import json
import os
import stat
import sys
import textwrap

import pytest

from ga_gemba.cli import main


def records(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def mbr_args(d, out="out.txt", *extra):
    return ["mbr", "-s", str(d / "src.txt"), "-t", str(d / "hyps.txt"), "--num-samples", "2",
            "-o", str(d / out), "--weights", "bleu=0.5,chrf=0.5", "-g", "5", *extra]


def gemba_adv_args(d, url, *extra):
    return ["adversarial", "-s", str(d / "src.txt"), "-t", str(d / "hyps.txt"), "-r", str(d / "ref.txt"),
            "--num-samples", "2", "-o", str(d / "adv.txt"), "-w", str(d / "words.txt"), "-g", "25",
            "--pop-multiplier", "10", "--seed", "1",
            "--target", "gemba_accuracy", "--penalize", "bleu,chrf", "--penalty", "0.1",
            "--gemba-model", "judge", "--gemba-base-url", url,
            "--src-lang-name", "English", "--tgt-lang-name", "Egyptian Arabic", *extra]


def test_mbr_writes_one_translation_per_source(corpus):
    main(mbr_args(corpus))
    lines = (corpus / "out.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    rec = records(corpus / "out.txt.jsonl")
    assert [r["index"] for r in rec] == [0, 1]
    assert rec[0]["best"]["fitness"] >= rec[0]["original"]["fitness"]
    assert set(rec[0]["best"]["scores"]) == {"bleu", "chrf"}


def test_mbr_with_zero_generations_reranks(corpus):
    main(mbr_args(corpus, "out.txt", "-g", "0"))
    rec = records(corpus / "out.txt.jsonl")[0]
    assert rec["best"]["text"] == rec["original"]["text"]
    assert rec["generations_run"] == 0


def test_adversarial_inflates_judge_score_and_reports_guards(corpus, judge_server):
    main(gemba_adv_args(corpus, judge_server.url))
    for rec in records(corpus / "adv.txt.jsonl"):
        assert rec["weights"] == {"gemba_accuracy": 1.0, "bleu": -0.1, "chrf": -0.1}
        assert rec["best"]["scores"]["gemba_accuracy"] > rec["original"]["scores"]["gemba_accuracy"]
        assert "good" in rec["best"]["text"]
        assert rec["refs"]
    assert judge_server.requests


def test_adversarial_minimize_reverses_all_signs(corpus, judge_server):
    main(gemba_adv_args(corpus, judge_server.url, "--direction", "minimize", "-g", "2"))
    assert records(corpus / "adv.txt.jsonl")[0]["weights"] == {"gemba_accuracy": -1.0, "bleu": 0.1, "chrf": 0.1}


def test_report_metrics_are_recorded_but_not_optimised(corpus):
    main(mbr_args(corpus, "out.txt", "--report-metrics", "chrf_multiref"))
    rec = records(corpus / "out.txt.jsonl")[0]
    assert "chrf_multiref" in rec["original"]["scores"] and "chrf_multiref" in rec["best"]["scores"]
    assert "chrf_multiref" not in rec["weights"]


def test_per_segment_target_languages(corpus, judge_server):
    (corpus / "langs.txt").write_text("Egyptian Arabic\nTunisian Arabic\n", encoding="utf-8")
    main(gemba_adv_args(corpus, judge_server.url, "--tgt-lang-file", str(corpus / "langs.txt"), "-g", "1"))
    prompts = [r["messages"][0]["content"] for r in judge_server.requests]
    assert any("Tunisian Arabic translation" in p for p in prompts)
    assert any("Egyptian Arabic translation" in p for p in prompts)


def test_shards_collect_to_the_same_result_as_one_run(corpus):
    main(mbr_args(corpus, "whole.txt"))
    for i in range(2):
        main(mbr_args(corpus, f"s{i}.txt", "--shard", f"{i}/2"))
    main(["collect", str(corpus / "s1.txt.jsonl"), str(corpus / "s0.txt.jsonl"), "-o", str(corpus / "merged.txt")])
    assert (corpus / "merged.txt").read_text() == (corpus / "whole.txt").read_text()


def test_collect_refuses_missing_shard(corpus, capsys):
    main(mbr_args(corpus, "s0.txt", "--shard", "0/2"))
    with pytest.raises(SystemExit):
        main(["collect", str(corpus / "s0.txt.jsonl"), "-o", str(corpus / "merged.txt")])
    assert "missing" in capsys.readouterr().err
    main(["collect", str(corpus / "s0.txt.jsonl"), "-o", str(corpus / "merged.txt"), "--allow-partial"])


def test_resume_continues_and_repairs_a_partial_write(corpus):
    main(mbr_args(corpus, "whole.txt"))
    whole = (corpus / "whole.txt").read_text()
    main(mbr_args(corpus, "out.txt"))
    # Simulate a job killed after segment 0, with half a JSON record on disk.
    jsonl = (corpus / "out.txt.jsonl").read_text().splitlines(True)
    (corpus / "out.txt.jsonl").write_text(jsonl[0] + jsonl[1][:20])
    (corpus / "out.txt").write_text(whole.splitlines(True)[0])
    main(mbr_args(corpus, "out.txt", "--resume"))
    assert (corpus / "out.txt").read_text() == whole
    assert [r["index"] for r in records(corpus / "out.txt.jsonl")] == [0, 1]


def test_existing_output_is_not_overwritten_without_resume(corpus, capsys):
    main(mbr_args(corpus))
    before = (corpus / "out.txt").read_text()
    with pytest.raises(SystemExit):
        main(mbr_args(corpus))
    assert "--resume" in capsys.readouterr().err
    assert (corpus / "out.txt").read_text() == before


def test_candidate_count_mismatch_is_an_error(corpus, capsys):
    (corpus / "hyps.txt").write_text("only one line\n")
    with pytest.raises(SystemExit):
        main(mbr_args(corpus))
    assert "expected 2 x 2" in capsys.readouterr().err


def test_remote_endpoint_requires_a_request_budget(corpus, capsys):
    with pytest.raises(SystemExit):
        main(gemba_adv_args(corpus, "https://api.example.com/v1"))
    assert "--gemba-max-requests" in capsys.readouterr().err


def test_budget_exhaustion_keeps_finished_segments_and_can_resume(corpus, judge_server, capsys):
    with pytest.raises(SystemExit):
        main(gemba_adv_args(corpus, judge_server.url, "-g", "1", "--gemba-max-requests", "3",
                            "--pop-multiplier", "2", "--no-empty"))
    assert "--gemba-max-requests" in capsys.readouterr().err
    assert not (corpus / "adv.txt").read_text() or (corpus / "adv.txt").read_text().endswith("\n")


def test_dead_endpoint_fails_instead_of_writing_output(corpus, capsys):
    url = "http://127.0.0.1:9/v1"
    with pytest.raises(SystemExit):
        main(gemba_adv_args(corpus, url, "-g", "1", "--pop-multiplier", "2"))
    assert "failed" in capsys.readouterr().err
    assert (corpus / "adv.txt").read_text() == ""


def test_gemba_metric_needs_a_model(corpus, capsys):
    args = gemba_adv_args(corpus, "http://127.0.0.1:1/v1")
    i = args.index("--gemba-model")
    del args[i : i + 2]
    with pytest.raises(SystemExit):
        main(args)
    assert "--gemba-model" in capsys.readouterr().err


def test_reference_metric_without_references_is_an_error(corpus, capsys):
    args = ["adversarial", "-s", str(corpus / "src.txt"), "-t", str(corpus / "hyps.txt"), "--num-samples", "2",
            "-o", str(corpus / "a.txt"), "--target", "bleu"]
    with pytest.raises(SystemExit):
        main(args)
    assert "-r/--refs" in capsys.readouterr().err


def fake_vllm(tmp_path, body):
    """An executable standing in for `vllm serve`."""
    path = tmp_path / "fake_vllm"
    path.write_text(f"#!{sys.executable}\n" + textwrap.dedent(body))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


SERVER_BODY = """
    import json, os, signal, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer
    args = sys.argv[1:]
    assert args[0] == "serve"
    open(os.environ["FAKE_VLLM_ARGS"], "w").write(json.dumps(args))
    open(os.environ["FAKE_VLLM_ARGS"] + ".env", "w").write(json.dumps(dict(os.environ)))
    port = int(args[args.index("--port") + 1])
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body)
        def do_GET(self): self._send({"data": []})
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self._send({"id": "x", "object": "chat.completion", "created": 0, "model": "m", "choices": [
                {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Score: 50"}}]})
    server = HTTPServer(("127.0.0.1", port), H)
    def stop(*a):
        open(os.environ["FAKE_VLLM_ARGS"] + ".stopped", "w").close()
        os._exit(0)
    signal.signal(signal.SIGTERM, stop)
    server.serve_forever()
"""


def test_managed_vllm_serves_the_run_and_is_stopped(corpus, monkeypatch):
    marker = corpus / "vllm_args.json"
    monkeypatch.setenv("FAKE_VLLM_ARGS", str(marker))
    monkeypatch.setenv("HF_HOME", "/somewhere/else")
    outer_home = os.environ["HOME"]
    exe = fake_vllm(corpus, SERVER_BODY)
    args = gemba_adv_args(corpus, "unused", "-g", "1", "--pop-multiplier", "2", "--gemba-backend", "vllm",
                          "--vllm-bin", str(exe), "--vllm-gpus", "2", "--vllm-max-model-len", "4096",
                          "--vllm-arg=--max-num-seqs=16", "--vllm-log", str(corpus / "v.log"))
    main(args)
    served = json.loads(marker.read_text())
    assert served[1] == "judge"
    assert served[served.index("--tensor-parallel-size") + 1] == "2"
    assert served[served.index("--max-model-len") + 1] == "4096"
    assert "--max-num-seqs=16" in served
    assert records(corpus / "adv.txt.jsonl")[0]["best"]["scores"]["gemba_accuracy"] == 0.5
    assert os.path.exists(str(marker) + ".stopped")
    env = json.loads(open(str(marker) + ".env").read())
    assert env["HOME"] == "/lnet/work/people/jon/llm_agents"
    assert "HF_HOME" not in env
    assert os.environ["HOME"] == outer_home  # COMET and the rest of the run keep their cache


def test_vllm_crash_at_startup_is_reported(corpus, capsys):
    exe = fake_vllm(corpus, "import sys; sys.exit(3)\n")
    with pytest.raises(SystemExit):
        main(gemba_adv_args(corpus, "unused", "--gemba-backend", "vllm", "--vllm-bin", str(exe),
                            "--vllm-log", str(corpus / "v.log")))
    assert "exited with code 3" in capsys.readouterr().err

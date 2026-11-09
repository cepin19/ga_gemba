import pytest

from ga_gemba.metrics import sacre
from ga_gemba.metrics.base import Metric, SegmentContext
from ga_gemba.metrics.gemba import GembaBudgetExceeded, GembaError, GembaMetric, Judge, JudgeConfig, parse_score
from ga_gemba.objective import Objective
from conftest import FakeServer


def ctx(**kw):
    base = dict(index=0, src="zdroj", refs=["a good ref"], src_lang="English", tgt_lang="Egyptian Arabic")
    return SegmentContext(**{**base, **kw})


@pytest.mark.parametrize("answer,expected", [
    ("Errors: none.\nScore: 85", 0.85),
    ("**Score:** 72.5", 0.725),
    ("Score: 90/100", 0.9),
    ("first guess Score: 30\nrevised Score: 60", 0.6),
    ("Score: 400\nScore: 40", 0.4),
    ("85", 0.85),
    ("85/100", 0.85),
    ("<think>Score: 1</think>Score: 77", 0.77),
    ("no number here", None),
    ("Score: 400", None),
    ("It has 3 errors and 2 omissions", None),
    ("", None),
])
def test_parse_score(answer, expected):
    assert parse_score(answer) == expected


@pytest.mark.parametrize("name", ["bleu", "bleu_multiref", "chrf", "chrf_multiref"])
def test_sacre_metrics_are_scaled_and_rank_sensibly(name):
    metric = sacre.make(name)
    good, bad = metric.score(ctx(refs=["the cat sat on the mat"]), ["the cat sat on the mat", "dogs run far away"])
    assert good == pytest.approx(1.0, abs=0.01) and bad < good


def test_mean_and_pooled_references_differ():
    refs = ["the cat sat", "dogs run fast"]
    hyp = ["the cat sat"]
    assert sacre.make("bleu").score(ctx(refs=refs), hyp)[0] < sacre.make("bleu_multiref").score(ctx(refs=refs), hyp)[0]


def test_candidate_does_not_match_itself_when_excluding():
    refs = ["the cat sat on the mat", "a dog ran across the road"]
    hyp = [refs[0]]
    metric = sacre.make("bleu")
    assert metric.score(ctx(refs=refs, exclude_identical_refs=True), hyp)[0] < 0.2
    assert metric.score(ctx(refs=refs, exclude_identical_refs=False), hyp)[0] > 0.5


def test_excluding_falls_back_when_every_reference_is_identical():
    refs = ["the same text", "the same text"]
    assert sacre.make("bleu").score(ctx(refs=refs, exclude_identical_refs=True), [refs[0]])[0] > 0.99


class Recording(Metric):
    needs_refs = False

    def __init__(self, values, ceiling=None):
        self.values, self.ceiling, self.calls = values, ceiling, []

    def score(self, context, hyps):
        self.calls.append(list(hyps))
        return [self.values[h] for h in hyps]


def test_objective_weights_each_metric_separately():
    a, b = Recording({"x": 0.5}), Recording({"x": 0.2})
    assert Objective({"a": a, "b": b}, {"a": 1.0, "b": -0.5}, ctx())(["x"]) == [pytest.approx(0.4)]


def test_objective_scores_each_distinct_text_once():
    metric = Recording({"x": 0.5, "y": 0.1})
    objective = Objective({"a": metric}, {"a": 1.0}, ctx())
    objective(["x", "x", "y"])
    objective(["y", "x"])
    assert metric.calls == [["x", "y"]]


def test_objective_failed_score_is_minus_infinity_even_with_negative_weight():
    metric = Recording({"x": None})
    assert Objective({"a": metric}, {"a": -1.0}, ctx())(["x"]) == [float("-inf")]


def test_objective_stops_only_for_a_single_positive_metric_with_ceiling():
    capped, free = Recording({}, ceiling=0.999), Recording({})
    assert Objective({"a": capped}, {"a": 2.0}, ctx()).stop_at == pytest.approx(1.998)
    assert Objective({"a": capped}, {"a": -1.0}, ctx()).stop_at is None
    assert Objective({"a": capped, "b": free}, {"a": 1.0, "b": 1.0}, ctx()).stop_at is None
    assert Objective({"b": free}, {"b": 1.0}, ctx()).stop_at is None


def make_judge(server, **kw):
    return Judge(JudgeConfig(base_url=server.url, model="m", **kw))


def test_dual_aspects_and_their_mean(judge_server):
    judge = make_judge(judge_server)
    hyps = ["good good yalla", "nothing"]
    assert GembaMetric(judge, "GEMBA-DA", "accuracy").score(ctx(), hyps) == [1.0, 0.2]
    assert GembaMetric(judge, "GEMBA-DA", "dialect").score(ctx(), hyps) == [1.0, 0.1]
    assert GembaMetric(judge, "GEMBA-DA", "both").score(ctx(), hyps) == [1.0, pytest.approx(0.15)]


def test_identical_prompts_are_shared_between_aspect_metrics(judge_server):
    judge = make_judge(judge_server)
    GembaMetric(judge, "GEMBA-DA", "both").score(ctx(), ["x"])
    GembaMetric(judge, "GEMBA-DA", "accuracy").score(ctx(), ["x"])
    assert judge.requests == 2


def test_prompt_contains_segment_and_languages(judge_server):
    GembaMetric(make_judge(judge_server), "GEMBA-DA", "accuracy").score(ctx(), ["hyp {with} braces"])
    prompt = judge_server.requests[0]["messages"][0]["content"]
    assert 'English source: "zdroj"' in prompt
    assert 'Egyptian Arabic translation: "hyp {with} braces"' in prompt


def test_reference_method_averages_over_references(judge_server):
    judge_server.reply = lambda p: "90" if "ref one" in p else "50"
    metric = GembaMetric(make_judge(judge_server), "GEMBA-DA_ref", "accuracy")
    assert metric.needs_refs
    assert metric.score(ctx(refs=["ref one", "ref two"]), ["h"]) == [pytest.approx(0.7)]


def test_unparsable_answer_is_unscored_after_retrying_with_sampling(judge_server):
    judge_server.reply = lambda p: "I refuse"
    judge = make_judge(judge_server, parse_retries=3)
    assert GembaMetric(judge, "GEMBA-DA", "accuracy").score(ctx(), ["h"]) == [None]
    assert [r["temperature"] for r in judge_server.requests] == [0.0, 0.7, 0.7]


def test_answer_recovered_on_retry(judge_server):
    answers = iter(["unclear", "Score: 66"])
    judge_server.reply = lambda p: next(answers)
    assert GembaMetric(make_judge(judge_server), "GEMBA-DA", "accuracy").score(ctx(), ["h"]) == [0.66]


def test_one_failed_aspect_leaves_the_candidate_unscored(judge_server):
    judge_server.reply = lambda p: "nothing" if "authentically" in p else "Score: 50"
    assert GembaMetric(make_judge(judge_server), "GEMBA-DA", "both").score(ctx(), ["h"]) == [None]


def test_judge_that_never_answers_usably_aborts(judge_server):
    judge_server.reply = lambda p: "I refuse"
    judge = make_judge(judge_server, parse_retries=1, max_consecutive_failures=3, concurrency=1)
    with pytest.raises(GembaError, match="consecutive"):
        GembaMetric(judge, "GEMBA-DA", "accuracy").score(ctx(), list("abcdef"))


def test_auth_failure_aborts(judge_server):
    judge_server.fail_with = 401
    with pytest.raises(GembaError, match="failed"):
        GembaMetric(make_judge(judge_server), "GEMBA-DA", "accuracy").score(ctx(), ["h"])


def test_rejected_prompt_only_loses_that_candidate(judge_server):
    judge_server.fail_with = 400
    assert GembaMetric(make_judge(judge_server), "GEMBA-DA", "accuracy").score(ctx(), ["h"]) == [None]


def test_request_budget_is_enforced(judge_server):
    judge = make_judge(judge_server, max_requests=2, concurrency=1)
    with pytest.raises(GembaBudgetExceeded):
        GembaMetric(judge, "GEMBA-DA", "accuracy").score(ctx(), list("abcd"))
    assert len(judge_server.requests) == 2


def test_gpt5_requests_omit_sampling_parameters():
    server = FakeServer()
    try:
        judge = Judge(JudgeConfig(base_url=server.url, model="gpt-5.4-nano"))
        GembaMetric(judge, "GEMBA-DA", "accuracy").score(ctx(), ["h"])
        assert "temperature" not in server.requests[0] and "max_tokens" not in server.requests[0]
    finally:
        server.close()


def test_api_key_comes_from_the_named_variable(judge_server, monkeypatch):
    monkeypatch.setenv("MY_JUDGE_KEY", "secret")
    GembaMetric(make_judge(judge_server, api_key_env="MY_JUDGE_KEY"), "GEMBA-DA", "accuracy").score(ctx(), ["h"])
    assert judge_server.auth[0] == "Bearer secret"


def test_aspect_metric_rejected_for_single_score_method(judge_server):
    with pytest.raises((ValueError, GembaError)):
        GembaMetric(make_judge(judge_server), "GEMBA-SQM", "accuracy")


WMT27 = "/lnet/work/people/jon/wmt27/final_rewriter/scripts"
needs_wmt27 = pytest.mark.skipif(not __import__("os").path.exists(f"{WMT27}/score_gemba.py"),
                                 reason="wmt27 score_gemba.py not available")


@needs_wmt27
def test_wmt27_da_accuracy_uses_its_prompt_and_strict_parser(judge_server):
    from ga_gemba.metrics.gemba import _load_wmt27
    wmt27 = _load_wmt27(WMT27)
    judge_server.reply = lambda p: "Score: 85" if "good" in p else "**Score:** 85"
    metric = GembaMetric(make_judge(judge_server, parse_retries=1), "wmt27-DA_accuracy", "both", WMT27)
    assert metric.score(ctx(), ["good", "bad"]) == [0.85, None]
    expected = wmt27.build({"source": "zdroj", "hyp": "good", "target_lang": "Egyptian Arabic", "method": "DA_accuracy"})
    assert expected in [r["messages"] for r in judge_server.requests]  # requests are concurrent
    assert len(judge_server.requests) == 2


@needs_wmt27
def test_wmt27_mqm_keeps_the_chat_transcript(judge_server):
    judge_server.reply = lambda p: "Critical:\nno-error\nMajor:\naccuracy/mistranslation - x\nMinor:\nno-error"
    metric = GembaMetric(make_judge(judge_server), "wmt27-MQM", "both", WMT27)
    assert metric.score(ctx(), ["h"]) == [pytest.approx(20 / 25)]
    assert len(judge_server.requests[0]["messages"]) > 1


@needs_wmt27
def test_wmt27_methods_reject_aspects(judge_server):
    with pytest.raises(ValueError):
        GembaMetric(make_judge(judge_server), "wmt27-DA_accuracy", "accuracy", WMT27)

"""Command line interface: ``mbr``, ``adversarial`` and ``collect``."""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from typing import Dict, List, Optional
from urllib.parse import urlparse

from tqdm import tqdm

from . import data
from .ga import GAConfig, GeneticAlgorithm, build_population, build_vocabulary, phrases_to_genes
from .metrics import GEMBA_ASPECTS, METRIC_NAMES, Metric, SegmentContext, make_metric
from .objective import Objective
from .tokenizers import make_tokenizer
from .vllm_server import DEFAULT_LLM_HOME

logger = logging.getLogger("ga_gemba")


def _names(spec: str) -> List[str]:
    return [n.strip() for n in spec.split(",") if n.strip()]


def _add_common(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("input and output")
    g.add_argument("-s", "--sources", required=True, help="One source segment per line")
    g.add_argument("-t", "--translations", required=True, help="--num-samples candidate lines per source")
    g.add_argument("--num-samples", type=int, required=True)
    g.add_argument("-o", "--output", required=True, help="Best translation per line")
    g.add_argument("--jsonl", help="Per-segment details (default: OUTPUT.jsonl)")
    g.add_argument("--resume", action="store_true", help="Continue after the segments already in the outputs")
    g.add_argument("--shard", help="Process only part I/N of the segments, I counted from 0")
    g.add_argument("--history", action="store_true", help="Store per-generation statistics in the JSONL")
    g.add_argument("--verbose", action="store_true")

    g = p.add_argument_group("objective")
    g.add_argument("--weights", help=f"name=weight,... Metrics: {', '.join(METRIC_NAMES)}")
    g.add_argument("--report-metrics", default="", help="Comma-separated metrics to record for the original and "
                   "final translation without optimising them")

    g = p.add_argument_group("genetic algorithm")
    g.add_argument("-g", "--generations", type=int, help="Breeding rounds; 0 only reranks the candidates")
    g.add_argument("-m", "--mutation", type=float, help="Expected mutated genes per candidate")
    g.add_argument("-c", "--crossover", type=float, help="Crossover probability per parent pair")
    g.add_argument("--pop-multiplier", type=int, help="Population size = num-samples x this")
    g.add_argument("--tokenizer", choices=["moses", "sp"], default="moses",
                   help="Genes: Moses words, or the flores200 pieces used by spBLEU")
    g.add_argument("--lang", default="en", help="Moses language code of the translations")
    g.add_argument("--no-empty", action="store_true", help="No empty slots: disables insertion and deletion")
    g.add_argument("-d", "--dict", help="Per-source mutation phrases, ';'-separated, one line per source")
    g.add_argument("-w", "--wordlist", help="Mutation phrases shared by all sources, ';'-separated")
    g.add_argument("--no-multitok-dict", action="store_true", help="Split dictionary phrases into separate genes")
    g.add_argument("--max-len-factor", type=float, help="Chromosome length = longest candidate x this")
    g.add_argument("--seed", type=int, default=0)

    g = p.add_argument_group("metric models")
    g.add_argument("--comet-model", default="Unbabel/wmt22-comet-da")
    g.add_argument("--comet-qe-model", default="Unbabel/wmt22-cometkiwi-da")
    g.add_argument("--bleurt-checkpoint")
    g.add_argument("--unite-model", default="damo/nlp_unite_mup_translation_evaluation_multilingual_base",
                   help="ModelScope model name or local snapshot directory")
    g.add_argument("--no-bleu-effective-order", action="store_true",
                   help="BLEU fitness without effective order, as in the LREC 2024 experiments")
    g.add_argument("--model-storage-path", help="COMET download directory")
    g.add_argument("--batch-size", type=int, default=64)

    g = p.add_argument_group("GEMBA (LLM judge)")
    g.add_argument("--gemba-method", default="GEMBA-DA", help="GEMBA-DA, GEMBA-DA_ref, an upstream method "
                   "(GEMBA-MQM, -SQM[_ref], -stars[_ref], -classes[_ref]), or wmt27-MQM, wmt27-DA_accuracy, "
                   "wmt27-DA_dialect (prompt and parser of the wmt27 score_gemba.py)")
    g.add_argument("--gemba-wmt27-scripts", default="/lnet/work/people/jon/wmt27/final_rewriter/scripts",
                   help="Directory with score_gemba.py, for the wmt27-* methods")
    g.add_argument("--gemba-model", help="Judge model name")
    g.add_argument("--gemba-backend", choices=["endpoint", "vllm"], default="endpoint",
                   help="endpoint: use a running OpenAI-style server; vllm: start vLLM for this run")
    g.add_argument("--gemba-base-url", default="https://api.openai.com/v1", help="Endpoint base URL")
    g.add_argument("--gemba-api-key-env", default="OPENAI_API_KEY",
                   help="Environment variable holding the API key; it is sent to the base URL")
    g.add_argument("--gemba-temperature", type=float, default=0.0)
    g.add_argument("--gemba-max-tokens", type=int, help="Default 500; 2048 for wmt27-* methods, as there")
    g.add_argument("--gemba-parse-retries", type=int,
                   help="Requests per prompt until an answer parses; default 3, 1 for wmt27-* methods, as there")
    g.add_argument("--gemba-concurrency", type=int, default=16)
    g.add_argument("--gemba-max-requests", type=int,
                   help="Abort after this many requests; -1 for no limit. Required for non-local endpoints")
    g.add_argument("--gemba-extra-body", type=json.loads, help="JSON merged into each request body")
    g.add_argument("--src-lang-name", help="Source language name used in the prompts, e.g. English")
    g.add_argument("--tgt-lang-name", help='Target language name used in the prompts, e.g. "Egyptian Arabic"')
    g.add_argument("--tgt-lang-file", help="Target language name per source line, overrides --tgt-lang-name")
    g.add_argument("--vllm-bin", default="vllm", help="vLLM executable, which may live in another environment")
    g.add_argument("--vllm-port", type=int)
    g.add_argument("--vllm-gpus", type=int, default=1)
    g.add_argument("--vllm-max-model-len", type=int)
    g.add_argument("--vllm-arg", action="append", default=[], help="Extra vLLM argument, e.g. --vllm-arg=--max-num-seqs=16")
    g.add_argument("--vllm-log", default="vllm.log")
    g.add_argument("--vllm-home", default=DEFAULT_LLM_HOME,
                   help="HOME of the vLLM process; models are stored in its .cache (default: %(default)s)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ga-gemba", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    mbr = commands.add_parser("mbr", help="Improve translations: maximise agreement with pseudo-references")
    _add_common(mbr)
    mbr.add_argument("-p", "--pseudo-refs", help="Pseudo-references (default: the candidates themselves)")
    mbr.add_argument("--num-pseudo-refs", type=int, help="Per source (default: --num-samples)")
    mbr.add_argument("--keep-identical-refs", action="store_true",
                     help="Let BLEU, chrF and GEMBA compare a candidate with an identical pseudo-reference")
    mbr.set_defaults(generations=200, mutation=1.0, crossover=0.1, pop_multiplier=15, max_len_factor=1.3,
                     weights="comet=0.4,comet_qe=0.4,bleu_multiref=0.1,chrf_multiref=0.1")

    adv = commands.add_parser("adversarial", help="Find translations on which metrics disagree")
    _add_common(adv)
    adv.add_argument("-r", "--refs", help="True references")
    adv.add_argument("--num-refs", type=int, default=1, help="Per source")
    adv.add_argument("--target", help="Metric to fool")
    adv.add_argument("--penalize", default="", help="Comma-separated guard metrics that should move the opposite way")
    adv.add_argument("--penalty", type=float, default=0.1, help="Weight magnitude of each guard metric")
    adv.add_argument("--direction", choices=["maximize", "minimize"], default="maximize",
                     help="maximize: target up, guards down (overrated output); minimize: the reverse")
    adv.set_defaults(generations=150, mutation=1.0, crossover=0.1, pop_multiplier=50, max_len_factor=1.1)

    col = commands.add_parser("collect", help="Merge shard outputs and print a summary")
    col.add_argument("jsonl", nargs="+", help="Shard JSONL files")
    col.add_argument("-o", "--output", help="Merged translations")
    col.add_argument("--merged-jsonl")
    col.add_argument("--allow-partial", action="store_true")
    return parser


def resolve_weights(args: argparse.Namespace, parser_error) -> Dict[str, float]:
    if args.command == "mbr" or args.weights:
        if args.command == "adversarial" and args.target:
            parser_error("Use either --weights or --target, not both")
        weights = data.parse_weights(args.weights)
    elif args.target:
        sign = 1.0 if args.direction == "maximize" else -1.0
        weights = {args.target: sign}
        for guard in _names(args.penalize):
            if guard == args.target:
                parser_error("--penalize cannot contain the --target metric")
            weights[guard] = -sign * args.penalty
    else:
        parser_error("adversarial needs --target or --weights")
    unknown = [n for n in weights if n not in METRIC_NAMES]
    if unknown:
        parser_error(f"Unknown metrics: {', '.join(unknown)}. Choose from: {', '.join(METRIC_NAMES)}")
    return weights


def _is_local(url: str) -> bool:
    return urlparse(url).hostname in ("localhost", "127.0.0.1", "::1")


def validate_gemba(args: argparse.Namespace, parser_error) -> None:
    if not args.gemba_model:
        parser_error("GEMBA metrics need --gemba-model")
    if args.gemba_backend == "endpoint" and not _is_local(args.gemba_base_url) and args.gemba_max_requests is None:
        parser_error(
            f"{args.gemba_base_url} is not local and a GA run can send a very large number of requests: "
            "set --gemba-max-requests (-1 for no limit)"
        )


def _make_judge(args: argparse.Namespace, base_url: str):
    from .metrics.gemba import Judge, JudgeConfig

    limit = args.gemba_max_requests
    wmt27 = args.gemba_method.startswith("wmt27-")
    max_tokens = args.gemba_max_tokens or (2048 if wmt27 else 500)
    retries = args.gemba_parse_retries or (1 if wmt27 else 3)
    return Judge(JudgeConfig(
        base_url=base_url, model=args.gemba_model, api_key_env=args.gemba_api_key_env,
        temperature=args.gemba_temperature, max_tokens=max_tokens, parse_retries=retries,
        concurrency=args.gemba_concurrency, max_requests=None if limit is None or limit < 0 else limit,
        extra_body=args.gemba_extra_body,
    ))


def _load_vocabulary_sources(args, n_segments: int, tokenizer):
    per_segment: List[List[str]] = [[] for _ in range(n_segments)]
    if args.dict:
        lines = data.read_lines(args.dict)
        if len(lines) != n_segments:
            raise ValueError(f"{args.dict}: expected {n_segments} lines, found {len(lines)}")
        per_segment = [data.split_phrases(line) for line in lines]
    shared: List[str] = []
    if args.wordlist:
        with open(args.wordlist, encoding="utf-8", errors="replace") as f:
            shared = phrases_to_genes(data.split_phrases(f.read()), tokenizer, not args.no_multitok_dict)
        shared = sorted(set(shared))
    return per_segment, shared


def _scored(text: str, fitness: float, scores: Dict) -> Dict:
    return {"text": text, "fitness": fitness, "scores": scores}


def run_ga(args: argparse.Namespace, weights: Dict[str, float], report: List[str], base_url: Optional[str]) -> None:
    sources = data.read_lines(args.sources)
    n = len(sources)
    candidates = data.read_grouped(args.translations, args.num_samples, n, "candidates")

    if args.command == "mbr":
        n_refs = args.num_pseudo_refs or args.num_samples
        refs = data.read_grouped(args.pseudo_refs or args.translations, n_refs, n, "pseudo-references")
        exclude_identical = not args.keep_identical_refs
    elif args.refs:
        refs = data.read_grouped(args.refs, args.num_refs, n, "references")
        exclude_identical = False
    else:
        refs = [[] for _ in range(n)]
        exclude_identical = False

    tgt_langs = data.read_lines(args.tgt_lang_file) if args.tgt_lang_file else [args.tgt_lang_name] * n
    if len(tgt_langs) != n:
        raise ValueError(f"{args.tgt_lang_file}: expected {n} lines, found {len(tgt_langs)}")

    tokenizer = make_tokenizer(args.tokenizer, args.lang)
    per_segment_phrases, shared_genes = _load_vocabulary_sources(args, n, tokenizer)

    judge = _make_judge(args, base_url) if base_url else None
    metrics: Dict[str, Metric] = {
        name: make_metric(name, args, lambda: judge) for name in dict.fromkeys(list(weights) + report)
    }
    needing_refs = [name for name, m in metrics.items() if m.needs_refs]
    if needing_refs and not refs[0]:
        raise ValueError(f"Metrics {', '.join(needing_refs)} need references: pass -r/--refs")

    jsonl = args.jsonl or args.output + ".jsonl"
    writer = data.OutputWriter(args.output, jsonl).open(args.resume)
    finished = writer.done_records()
    indices = list(data.shard_range(n, *data.parse_shard(args.shard)))[writer.n_done :]
    if writer.n_done:
        logger.info("Resuming: %d segments already finished", writer.n_done)

    try:
        for index in tqdm(indices, desc="segments", disable=not sys.stderr.isatty()):
            context = SegmentContext(
                index=index, src=sources[index], refs=refs[index], src_lang=args.src_lang_name,
                tgt_lang=tgt_langs[index], exclude_identical_refs=exclude_identical,
            )
            objective = Objective(metrics, weights, context)
            config = GAConfig(
                generations=args.generations, mutation_rate=args.mutation, crossover_rate=args.crossover,
                allow_deletions=not args.no_empty, seed=args.seed + index,
            )
            hyps = candidates[index]
            vocabulary = build_vocabulary(
                hyps, tokenizer, per_segment_phrases[index], not args.no_multitok_dict, shared_genes
            )
            population = build_population(hyps, tokenizer, args.pop_multiplier, not args.no_empty, args.max_len_factor)
            ga = GeneticAlgorithm(objective, tokenizer, vocabulary, config, stop_at=objective.stop_at)
            result = ga.run(population)

            original, best = result.initial_best.text, result.best.text
            extra = {
                name: metrics[name].score(context, [original, best]) for name in report if name not in weights
            }
            record = {
                "index": index,
                "total": n,
                "src": context.src,
                "weights": weights,
                "original": _scored(original, result.initial_best_fitness,
                                    {**objective.scores(original), **{k: v[0] for k, v in extra.items()}}),
                "best": _scored(best, result.best_fitness,
                                {**objective.scores(best), **{k: v[1] for k, v in extra.items()}}),
                "generations_run": result.generations_run,
            }
            if args.command == "adversarial" and refs[index]:
                record["refs"] = refs[index]
            if args.history:
                record["history"] = result.history
            writer.write(best, record)
            finished.append(record)
    finally:
        writer.close()
        if judge:
            logger.info("Judge requests sent: %d", judge.requests)
    print(data.summarize(finished), file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "openai", "httpcore"):  # one log line per judge request otherwise
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.command == "collect":
        try:
            records = data.collect(args.jsonl, args.output, args.merged_jsonl, args.allow_partial)
        except ValueError as e:
            parser.exit(1, f"error: {e}\n")
        print(data.summarize(records), file=sys.stderr)
        return

    weights = resolve_weights(args, parser.error)
    report = _names(args.report_metrics)
    unknown = [m for m in report if m not in METRIC_NAMES]
    if unknown:
        parser.error(f"Unknown metrics: {', '.join(unknown)}")
    uses_gemba = any(name in GEMBA_ASPECTS for name in list(weights) + report)
    if uses_gemba:
        validate_gemba(args, parser.error)

    if uses_gemba and args.gemba_backend == "vllm":
        from .vllm_server import managed_vllm

        server = managed_vllm(
            args.gemba_model, args.vllm_bin, args.vllm_port, args.vllm_gpus,
            args.vllm_max_model_len, args.vllm_arg, args.vllm_log, home=args.vllm_home,
        )
    elif uses_gemba:
        server = contextlib.nullcontext(args.gemba_base_url)
    else:
        server = contextlib.nullcontext(None)

    try:
        with server as base_url:
            run_ga(args, weights, report, base_url)
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError) as e:
        # GembaError is a RuntimeError. Finished segments are on disk, so --resume continues.
        parser.exit(1, f"error: {e}\n")

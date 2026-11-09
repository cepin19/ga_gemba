"""GEMBA: an LLM as the judge, through any OpenAI-compatible chat endpoint.

The GEMBA-DA and GEMBA-DA_ref templates score translation accuracy and dialect
authenticity separately. The other methods take prompts and parsers from the upstream
``gemba`` package, which is imported only if one of them is requested.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from .base import Metric, SegmentContext

logger = logging.getLogger(__name__)

TEMPLATE_ACCURACY = (
    "Score the following translation from {source_lang} to {target_lang} on a continuous "
    "scale from 0 to 100 for translation accuracy only (meaning preservation and grammar). "
    "You can briefly reason about the errors and describe them, but make sure to output a "
    '"Score:" and a single number as the final score at the end of the output.\n\n'
    '{source_lang} source: "{source_seg}"\n'
    '{target_lang} translation: "{target_seg}"\n'
    "Evaluation:"
)
TEMPLATE_DIALECT = (
    "Score how authentically the following translation uses {target_lang} (appropriate "
    "dialect, register, and locale conventions) on a scale from 0 to 100, where 0 means "
    '"not authentic / wrong variety" and 100 means "fully authentic {target_lang}". '
    "If a dialect is specified and the translation is entirely in a different dialect, "
    "score the translation 0. "
    'You can briefly reason about the errors and describe them, but make sure to output a "Score:" '
    "and a single number as the final score at the end of the output.\n\n"
    '{source_lang} source: "{source_seg}"\n'
    '{target_lang} translation: "{target_seg}"\n'
    "Evaluation: "
)
TEMPLATE_ACCURACY_REF = (
    "Score the following machine translation from {source_lang} to {target_lang} for "
    "translation accuracy only (meaning vs. the source and human reference) on a scale "
    "from 0 to 100. Ignore dialect/register variety.\n\n"
    '{source_lang} source: "{source_seg}"\n'
    '{target_lang} human reference: "{reference_seg}"\n'
    '{target_lang} machine translation: "{target_seg}"\n'
    "Score: "
)
TEMPLATE_DIALECT_REF = (
    "Score how authentically the machine translation uses {target_lang} compared with "
    "the expected variety on a scale from 0 to 100. Ignore meaning accuracy.\n\n"
    '{source_lang} source: "{source_seg}"\n'
    '{target_lang} human reference: "{reference_seg}"\n'
    '{target_lang} machine translation: "{target_seg}"\n'
    "Score: "
)

DUAL_TEMPLATES = {
    "GEMBA-DA": {"accuracy": TEMPLATE_ACCURACY, "dialect": TEMPLATE_DIALECT},
    "GEMBA-DA_ref": {"accuracy": TEMPLATE_ACCURACY_REF, "dialect": TEMPLATE_DIALECT_REF},
}
UPSTREAM_METHODS = (
    "GEMBA-MQM", "GEMBA-SQM", "GEMBA-SQM_ref", "GEMBA-stars", "GEMBA-stars_ref",
    "GEMBA-classes", "GEMBA-classes_ref",
)
METHODS = tuple(DUAL_TEMPLATES) + UPSTREAM_METHODS
# "wmt27-<name>" takes prompt and parser from score_gemba.py of the wmt27 rewriter experiments.
WMT27_PREFIX = "wmt27-"
WMT27_SCRIPTS = "/lnet/work/people/jon/wmt27/final_rewriter/scripts"
WMT27_METHODS = ("MQM", "DA_accuracy", "DA_dialect")

_SCORE_RE = re.compile(r"score\W{0,6}?(\d+(?:\.\d+)?)", re.IGNORECASE)
_BARE_RE = re.compile(r"^\W*(\d+(?:\.\d+)?)(?:\s*/\s*100)?\W*$")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_score(answer: str) -> Optional[float]:
    """Return the last in-range "Score: N" as a 0-1 value, or None.

    A bare number also counts, because the *_ref templates end in "Score: " and
    models often just continue that with the number.
    """
    answer = _THINK_RE.sub("", answer).strip()
    for found in reversed(_SCORE_RE.findall(answer)):
        value = float(found)
        if 0 <= value <= 100:
            return value / 100
    bare = _BARE_RE.match(answer)
    if bare and 0 <= float(bare.group(1)) <= 100:
        return float(bare.group(1)) / 100
    return None


# A prompt is a single user message or, for chat templates, a tuple of (role, content) turns.
Prompt = Union[str, Tuple[Tuple[str, str], ...]]


def _messages(prompt: Prompt) -> List[dict]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return [{"role": role, "content": content} for role, content in prompt]


class GembaError(RuntimeError):
    """The judge cannot be used, so continuing would only produce meaningless output."""


class GembaBudgetExceeded(GembaError):
    pass


@dataclass
class JudgeConfig:
    base_url: str
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 500
    concurrency: int = 16
    parse_retries: int = 3
    max_requests: Optional[int] = None
    max_consecutive_failures: int = 50
    timeout: float = 600.0
    extra_body: Optional[dict] = None


# These reject temperature and max_tokens.
_FIXED_SAMPLING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


class Judge:
    """Sends prompts concurrently. Identical prompts are asked once per segment."""

    def __init__(self, config: JudgeConfig):
        from openai import OpenAI

        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=os.environ.get(config.api_key_env) or "EMPTY",
            timeout=config.timeout,
            max_retries=5,  # connection errors, 429 and 5xx, with backoff
        )
        self.requests = 0
        self._consecutive_failures = 0
        self._lock = threading.Lock()
        self._memo: Dict[Prompt, Optional[float]] = {}
        self._segment: Optional[int] = None

    def _count_request(self) -> None:
        with self._lock:
            limit = self.config.max_requests
            if limit is not None and self.requests >= limit:
                raise GembaBudgetExceeded(f"Reached --gemba-max-requests={limit}")
            self.requests += 1

    def _complete(self, prompt: "Prompt", temperature: float) -> str:
        cfg = self.config
        kwargs: dict = {}
        if not cfg.model.startswith(_FIXED_SAMPLING_PREFIXES):
            kwargs.update(temperature=temperature, max_tokens=cfg.max_tokens)
        if cfg.extra_body:
            kwargs["extra_body"] = cfg.extra_body
        self._count_request()
        response = self.client.chat.completions.create(
            model=cfg.model, messages=_messages(prompt), **kwargs
        )
        if not response.choices:
            return ""
        return response.choices[0].message.content or ""

    def _ask(self, prompt: Prompt, parse: Callable[[str], Optional[float]]) -> Optional[float]:
        import openai

        cfg = self.config
        answer = ""
        for attempt in range(cfg.parse_retries):
            # At temperature 0 asking again would return the same unparsable answer.
            temperature = cfg.temperature if attempt == 0 else max(cfg.temperature, 0.7)
            try:
                answer = self._complete(prompt, temperature)
            except openai.BadRequestError as e:  # e.g. this candidate exceeds the context size
                logger.warning("Judge rejected a prompt: %s", e)
                break
            except openai.OpenAIError as e:
                raise GembaError(f"Request to {cfg.base_url} failed: {e}") from e
            score = parse(answer)
            if score is not None:
                with self._lock:
                    self._consecutive_failures = 0
                return score
        logger.warning("No usable score from judge. Last answer: %.200r", answer)
        with self._lock:
            self._consecutive_failures += 1
            failures = self._consecutive_failures
        if failures >= cfg.max_consecutive_failures:
            raise GembaError(
                f"{failures} consecutive prompts produced no usable score from {cfg.model} at "
                f"{cfg.base_url}. Last answer: {answer[:300]!r}"
            )
        return None

    def ask_many(
        self, segment: int, jobs: Sequence[Tuple[Prompt, Callable[[str], Optional[float]]]]
    ) -> List[Optional[float]]:
        if segment != self._segment:
            self._memo.clear()
            self._segment = segment
        unique = {prompt: parse for prompt, parse in jobs if prompt not in self._memo}
        if unique:
            with ThreadPoolExecutor(max_workers=self.config.concurrency) as pool:
                results = list(pool.map(lambda item: self._ask(*item), unique.items()))
            self._memo.update(zip(unique, results))
        return [self._memo[prompt] for prompt, _ in jobs]


def _upstream(method: str):
    """Prompt template and 0-1 parser for a method defined by the upstream gemba package."""
    try:
        from gemba.gemba_mqm_utils import TEMPLATE_GEMBA_MQM, parse_mqm_answer
        from gemba.prompt import prompts
    except ImportError as e:
        raise GembaError(f"{method} needs the upstream 'gemba' package: {e}") from e

    if method == "GEMBA-MQM":
        def parse(answer):
            value = parse_mqm_answer(answer, list_mqm_errors=False, full_desc=True)
            return max(0.0, min(1.0, value / 100)) if isinstance(value, (int, float)) else None
        return TEMPLATE_GEMBA_MQM, parse

    validate = prompts[method]["validate_answer"]
    if "stars" in method:  # 1-5
        lowest, span = 1, 4
    elif "classes" in method:  # index 0-4
        lowest, span = 0, 4
    else:  # 0-100
        lowest, span = 0, 100

    def parse(answer):
        value = validate(answer)
        return (value - lowest) / span if value is not None else None
    return prompts[method]["prompt"], parse


def _load_wmt27(scripts_dir: str):
    """Import score_gemba.py from the wmt27 experiments.

    It imports templates from the ``gemba`` and ``aramt`` packages, whose ``__init__`` modules pull in
    many unrelated dependencies (pandas, absl, mt_metrics_eval, ...). Those packages are registered
    here as bare packages, so only the modules score_gemba.py needs are executed.
    """
    import importlib.util
    import sys
    import types

    path = os.path.join(scripts_dir, "score_gemba.py")
    name = "_wmt27_score_gemba_" + str(abs(hash(os.path.abspath(path))))
    if name in sys.modules:
        return sys.modules[name]
    try:
        with open(path, encoding="utf-8") as f:
            student = re.search(r'^STUDENT = "(.+)"', f.read(), re.M).group(1)
    except (OSError, AttributeError) as e:
        raise GembaError(f"Cannot read STUDENT from {path}: {e}") from e
    for package, directory in (("gemba", f"{student}/GEMBA/gemba"), ("aramt", f"{student}/aramt"),
                               ("aramt.evaluation", f"{student}/aramt/evaluation")):
        if package not in sys.modules:
            module = types.ModuleType(package)
            module.__path__ = [directory]
            sys.modules[package] = module
    if importlib.util.find_spec("ipdb") is None:  # imported by gemba_mqm_utils but never used there
        sys.modules["ipdb"] = types.ModuleType("ipdb")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise GembaError(f"Cannot import {path}: {e}") from e
    sys.modules[name] = module
    return module


def _wmt27(method: str, scripts_dir: str):
    """Template, 0-1 parser and template filler of a wmt27 method."""
    name = method[len(WMT27_PREFIX):]
    if name not in WMT27_METHODS:
        raise ValueError(f"Unknown wmt27 GEMBA method {name}. Choose from: {', '.join(WMT27_METHODS)}")
    module = _load_wmt27(scripts_dir)
    template, raw_parse = module.METHODS[name]
    if name == "MQM":  # penalty, -25 (worst) to 0
        def parse(answer):
            value = raw_parse(answer)
            return (value + 25) / 25 if value is not None else None
    else:  # 0-100
        def parse(answer):
            value = raw_parse(answer)
            return value / 100 if value is not None else None

    def fill(template, **fields) -> Prompt:
        filled = module.apply_template(template, {
            k: fields[k] for k in ("source_lang", "target_lang", "source_seg", "target_seg")})
        if isinstance(filled, str):
            return filled
        return tuple((turn["role"], turn["content"]) for turn in filled)
    return template, parse, fill


def _fill(template, **fields) -> str:
    if isinstance(template, str):
        return template.format(**fields)
    # Upstream MQM templates are chat transcripts; flatten them to one user message.
    return "\n\n".join(turn["content"].format(**fields) for turn in template)


class GembaMetric(Metric):
    """aspect: 'accuracy', 'dialect' or 'both' (their mean); ignored by non-dual methods."""

    def __init__(self, judge: Judge, method: str, aspect: str = "both", wmt27_scripts: str = WMT27_SCRIPTS):
        self.fill = _fill
        if method.startswith(WMT27_PREFIX):
            if aspect != "both":
                raise ValueError(f"wmt27 methods give one score: use the metric gemba, not gemba_{aspect}")
            self.judge, self.method, self.needs_refs = judge, method, False
            template, parse, self.fill = _wmt27(method, wmt27_scripts)
            self.prompts = [(template, parse)]
            return
        if method not in METHODS:
            raise ValueError(f"Unknown GEMBA method {method}. Choose from: {', '.join(METHODS)}")
        self.judge = judge
        self.method = method
        self.needs_refs = method.endswith("_ref")
        if method in DUAL_TEMPLATES:
            aspects = ("accuracy", "dialect") if aspect == "both" else (aspect,)
            self.prompts = [(DUAL_TEMPLATES[method][a], parse_score) for a in aspects]
        elif aspect != "both":
            raise ValueError(f"gemba_{aspect} needs a dual method (GEMBA-DA or GEMBA-DA_ref), not {method}")
        else:
            self.prompts = [_upstream(method)]

    def score(self, ctx: SegmentContext, hyps: Sequence[str]) -> List[Optional[float]]:
        if not ctx.src_lang or not ctx.tgt_lang:
            raise GembaError("GEMBA needs --src-lang-name and --tgt-lang-name (or --tgt-lang-file)")
        jobs = []
        owners = []
        for h, hyp in enumerate(hyps):
            refs = ctx.refs_for(hyp) if self.needs_refs else [""]
            for ref in refs:
                for template, parse in self.prompts:
                    prompt = self.fill(
                        template, source_lang=ctx.src_lang, target_lang=ctx.tgt_lang,
                        source_seg=ctx.src, target_seg=hyp, reference_seg=ref,
                    )
                    jobs.append((prompt, parse))
                    owners.append(h)
        collected: List[List[Optional[float]]] = [[] for _ in hyps]
        for owner, value in zip(owners, self.judge.ask_many(ctx.index, jobs)):
            collected[owner].append(value)
        return [None if None in values else sum(values) / len(values) for values in collected]

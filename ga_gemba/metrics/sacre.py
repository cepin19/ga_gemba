"""sacreBLEU-based metrics. Plain names average over references; *_multiref pools them."""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from .base import Metric, SegmentContext


class SacreMetric(Metric):
    ceiling = 0.999

    def __init__(self, scorer, pool_refs: bool):
        self.scorer = scorer
        self.pool_refs = pool_refs

    def score(self, ctx: SegmentContext, hyps: Sequence[str]) -> List[Optional[float]]:
        return [self._score_one(h, ctx.refs_for(h)) for h in hyps]

    def _score_one(self, hyp: str, refs: List[str]) -> float:
        if self.pool_refs:
            return self.scorer.sentence_score(hyp, refs).score / 100
        return sum(self.scorer.sentence_score(hyp, [r]).score for r in refs) / len(refs) / 100


def make(name: str, effective_order: bool = True) -> SacreMetric:
    from sacrebleu.metrics import BLEU, CHRF

    if not effective_order:  # sacrebleu warns on every sentence otherwise
        logging.getLogger("sacrebleu").setLevel(logging.ERROR)

    base, _, suffix = name.partition("_")
    if base == "bleu":
        scorer = BLEU(effective_order=effective_order)
    elif base == "spbleu":
        scorer = BLEU(tokenize="flores200", effective_order=effective_order)
    elif base == "chrf":
        scorer = CHRF()
    else:
        raise ValueError(f"Unknown sacreBLEU metric: {name}")
    return SacreMetric(scorer, pool_refs=suffix == "multiref")

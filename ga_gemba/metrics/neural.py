"""BLEURT and UniTE, averaged over references."""
from __future__ import annotations

from typing import List, Optional, Sequence

from .base import Metric, SegmentContext

UNITE_MODEL = "damo/nlp_unite_mup_translation_evaluation_multilingual_base"


class Bleurt(Metric):
    def __init__(self, checkpoint: str, batch_size: int = 64):
        try:
            # TensorFlow loads system CUDA libraries that break a later `import torch`.
            import torch  # noqa: F401
        except ImportError:
            pass
        from bleurt import score as bleurt_score

        self.scorer = bleurt_score.BleurtScorer(checkpoint)
        self.batch_size = batch_size

    def score(self, ctx: SegmentContext, hyps: Sequence[str]) -> List[Optional[float]]:
        totals = [0.0] * len(hyps)
        for ref in ctx.refs:
            scores = self.scorer.score(references=[ref] * len(hyps), candidates=list(hyps), batch_size=self.batch_size)
            totals = [t + float(s) for t, s in zip(totals, scores)]
        return [t / len(ctx.refs) for t in totals]


class Unite(Metric):
    def __init__(self, batch_size: int = 64, model: str = UNITE_MODEL):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self.pipe = pipeline(task=Tasks.translation_evaluation, model=model)
        self.batch_size = batch_size

    def score(self, ctx: SegmentContext, hyps: Sequence[str]) -> List[Optional[float]]:
        totals = [0.0] * len(hyps)
        for ref in ctx.refs:
            for start in range(0, len(hyps), self.batch_size):
                chunk = list(hyps[start : start + self.batch_size])
                out = self.pipe({"src": [ctx.src] * len(chunk), "hyp": chunk, "ref": [ref] * len(chunk)})
                for i, s in enumerate(out["score"]):
                    totals[start + i] += float(s)
        return [t / len(ctx.refs) for t in totals]

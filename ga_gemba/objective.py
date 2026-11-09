"""Weighted fitness over several metrics, memoised per source segment."""
from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence

from .metrics.base import Metric, SegmentContext


class Objective:
    """fitness(text) = sum(weight * metric(text)); every metric is scaled to about [0, 1].

    Negative weights push a metric down, which is how adversarial runs are expressed.
    A candidate that any metric failed to score gets -inf and can never win. Each
    distinct text is scored once, so caching spends no accuracy for deterministic
    metrics and keeps a single stable value for sampled LLM judges.
    """

    def __init__(self, metrics: Mapping[str, Metric], weights: Mapping[str, float], context: SegmentContext):
        self.metrics = metrics
        self.weights = {k: v for k, v in weights.items() if v != 0}
        if not self.weights:
            raise ValueError("At least one metric needs a non-zero weight")
        self.context = context
        self._scores: Dict[str, Dict[str, Optional[float]]] = {}

    def __call__(self, texts: Sequence[str]) -> List[float]:
        new = [t for t in dict.fromkeys(texts) if t not in self._scores]
        if new:
            per_metric = {n: self.metrics[n].score(self.context, new) for n in self.weights}
            for i, text in enumerate(new):
                self._scores[text] = {n: per_metric[n][i] for n in self.weights}
        return [self._fitness(self._scores[t]) for t in texts]

    def _fitness(self, scores: Mapping[str, Optional[float]]) -> float:
        if any(scores[n] is None for n in self.weights):
            return -math.inf
        return sum(w * scores[n] for n, w in self.weights.items())

    def scores(self, text: str) -> Dict[str, Optional[float]]:
        self([text])
        return dict(self._scores[text])

    @property
    def stop_at(self) -> Optional[float]:
        """Fitness at which searching is pointless: only for one metric with a known maximum."""
        if len(self.weights) != 1:
            return None
        name, weight = next(iter(self.weights.items()))
        ceiling = self.metrics[name].ceiling
        return ceiling * weight if ceiling is not None and weight > 0 else None

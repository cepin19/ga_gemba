"""Metric registry. Heavy dependencies are imported only for the metrics actually used."""
from __future__ import annotations

from .base import Metric, SegmentContext

SACRE = ("bleu", "bleu_multiref", "chrf", "chrf_multiref", "spbleu", "spbleu_multiref")
GEMBA_ASPECTS = {"gemba": "both", "gemba_accuracy": "accuracy", "gemba_dialect": "dialect"}
METRIC_NAMES = SACRE + ("comet", "comet_qe", "bleurt", "unite") + tuple(GEMBA_ASPECTS)

__all__ = ["Metric", "SegmentContext", "METRIC_NAMES", "GEMBA_ASPECTS", "make_metric"]


def make_metric(name: str, args, get_judge) -> Metric:
    """Build a metric from CLI options. ``get_judge`` returns the shared LLM judge."""
    if name in SACRE:
        from . import sacre

        return sacre.make(name, effective_order=not args.no_bleu_effective_order)
    if name == "comet":
        from .comet import CometRef

        return CometRef(args.comet_model, args.model_storage_path, args.batch_size)
    if name == "comet_qe":
        from .comet import CometQE

        return CometQE(args.comet_qe_model, args.model_storage_path, args.batch_size)
    if name == "bleurt":
        from .neural import Bleurt

        if not args.bleurt_checkpoint:
            raise ValueError("Metric bleurt needs --bleurt-checkpoint")
        return Bleurt(args.bleurt_checkpoint, args.batch_size)
    if name == "unite":
        from .neural import Unite

        return Unite(args.batch_size, args.unite_model)
    if name in GEMBA_ASPECTS:
        from .gemba import GembaMetric

        return GembaMetric(get_judge(), args.gemba_method, GEMBA_ASPECTS[name], args.gemba_wmt27_scripts)
    raise ValueError(f"Unknown metric {name}. Choose from: {', '.join(METRIC_NAMES)}")

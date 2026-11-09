# Parts of this file derive from the MBR command of Unbabel COMET:
# Copyright (C) 2020 Unbabel, licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0).
"""COMET metrics.

``comet`` scores a hypothesis against each reference and averages: MBR decoding with
pseudo-references, ordinary reference-based COMET with a single true reference.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence

from .base import Metric, SegmentContext

logger = logging.getLogger(__name__)


def _load(name: str, storage: Optional[str]):
    import torch
    from comet.models import download_model, load_from_checkpoint

    path = name if name.endswith(".ckpt") and os.path.exists(name) else download_model(name, saving_directory=storage)
    model = load_from_checkpoint(path)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    else:
        logger.warning("No GPU available: running COMET model %s on CPU", name)
    return model


class CometRef(Metric):
    def __init__(self, model_name: str, storage: Optional[str] = None, batch_size: int = 64):
        self.model = _load(model_name, storage)
        self.batch_size = min(batch_size, 250)
        self._embeddings: Dict[str, object] = {}
        self._segment: Optional[int] = None

    def _embed(self, texts: Sequence[str]):
        import torch

        missing = [t for t in dict.fromkeys(texts) if t not in self._embeddings]
        with torch.no_grad():
            for i in range(0, len(missing), self.batch_size):
                chunk = missing[i : i + self.batch_size]
                batch = self.model.encoder.prepare_sample(chunk)
                embedded = self.model.get_sentence_embedding(
                    batch["input_ids"].to(self.model.device),
                    batch["attention_mask"].to(self.model.device),
                )
                self._embeddings.update(zip(chunk, embedded))
        return torch.stack([self._embeddings[t] for t in texts])

    def score(self, ctx: SegmentContext, hyps: Sequence[str]) -> List[Optional[float]]:
        import torch

        if ctx.index != self._segment:  # embeddings are only reused within a segment
            self._embeddings.clear()
            self._segment = ctx.index
        n_refs = len(ctx.refs)
        src = self._embed([ctx.src]).repeat(n_refs, 1)
        refs = self._embed(ctx.refs)
        mts = self._embed(hyps)
        scores: List[Optional[float]] = []
        with torch.no_grad():
            for mt in mts:
                result = self.model.estimate(src, mt.repeat(n_refs, 1), refs)
                scores.append(float(result["score"].mean()))
        return scores


class CometQE(Metric):
    needs_refs = False

    def __init__(self, model_name: str, storage: Optional[str] = None, batch_size: int = 64):
        import torch

        self.model = _load(model_name, storage)
        self.batch_size = batch_size
        self.gpus = 1 if torch.cuda.is_available() else 0

    def score(self, ctx: SegmentContext, hyps: Sequence[str]) -> List[Optional[float]]:
        samples = [{"src": ctx.src, "mt": h} for h in hyps]
        output = self.model.predict(
            samples=samples, batch_size=self.batch_size, gpus=self.gpus, accelerator="auto",
            length_batching=True, progress_bar=False,
        )
        return [float(s) for s in output.scores]

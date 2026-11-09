from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class SegmentContext:
    index: int
    src: str
    refs: List[str]  # true references (adversarial) or pseudo-references (MBR)
    src_lang: Optional[str] = None
    tgt_lang: Optional[str] = None
    # In MBR the pseudo-references are the candidates, so a hypothesis must not be
    # compared with itself. True references are always kept.
    exclude_identical_refs: bool = False

    def refs_for(self, hyp: str) -> List[str]:
        if self.exclude_identical_refs and len(self.refs) > 1:
            kept = [r for r in self.refs if r != hyp]
            if kept:
                return kept
        return self.refs


class Metric:
    """Scores hypotheses for one segment, on a scale of about [0, 1]; None means unscored."""

    needs_refs = True
    ceiling: Optional[float] = None  # score at which further search cannot help

    def score(self, ctx: SegmentContext, hyps: Sequence[str]) -> List[Optional[float]]:
        raise NotImplementedError

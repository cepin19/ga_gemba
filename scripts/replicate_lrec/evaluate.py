#!/usr/bin/env python3
"""Score output files the way Table 1 was scored: first N lines, corpus chrF and BLEU,
mean segment scores for the neural metrics. Appends TSV rows: name, then seven scores.

    evaluate.py SRC REF BLEURT_CKPT OUT.tsv N FILE...
"""
import os
import sys

import sacrebleu

from ga_gemba.metrics.comet import CometQE, CometRef
from ga_gemba.metrics.neural import Bleurt, Unite
from ga_gemba.metrics.base import SegmentContext


def lines(path, n):
    with open(path, encoding="utf-8", errors="replace") as f:
        return [l.strip() for l in f][:n]


def per_segment(metric, src, ref, hyp):
    total = 0.0
    for i, (s, r, h) in enumerate(zip(src, ref, hyp)):
        total += metric.score(SegmentContext(i, s, [r]), [h or "EMPTY"])[0]
    return total / len(hyp)


def main():
    src_path, ref_path, ckpt, out, n = sys.argv[1:6]
    n = int(n)
    src, ref = lines(src_path, n), lines(ref_path, n)
    metrics = [
        ("CMT20", CometRef("Unbabel/wmt20-comet-da")), ("CMT22", CometRef("Unbabel/wmt22-comet-da")),
        ("QE", CometQE("Unbabel/wmt22-cometkiwi-da")), ("BLEURT", Bleurt(ckpt)), ("UniTE", Unite(model=os.environ["UNITE_MODEL"])),
    ]
    with open(out, "a", encoding="utf-8") as f:
        for path in sys.argv[6:]:
            hyp = lines(path, n)
            if len(hyp) < n:
                print(f"SKIP {path}: {len(hyp)} < {n} lines", file=sys.stderr)
                continue
            row = [sacrebleu.corpus_chrf(hyp, [ref]).score, sacrebleu.corpus_bleu(hyp, [ref]).score]
            row += [per_segment(m, src, ref, hyp) for _, m in metrics]
            name = os.path.basename(path).rsplit(".", 1)[0]
            f.write("\t".join([name] + [f"{v:.4f}" for v in row]) + "\n")
            f.flush()
            print(name, row, flush=True)


if __name__ == "__main__":
    main()

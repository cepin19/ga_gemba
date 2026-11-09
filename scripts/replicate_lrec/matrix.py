#!/usr/bin/env python3
"""The 49 runs of Table 1 in Jon & Bojar, GAATME (LREC-COLING 2024).

Print one shell-quoted run per line as `NAME ARGS...`. BLEURT and CMT20 are never
controls: the CMT20 and BLEURT rows optimise them against the same controls as the rest.
"""
import shlex

CONTROLS = ["chrf_multiref", "bleu_multiref", "comet", "comet_qe", "unite"]
TARGETS = {
    "chrF": "chrf_multiref", "BLEU": "bleu_multiref", "CMT20": "comet", "CMT22": "comet",
    "CMT22-QE": "comet_qe", "BLEURT": "bleurt", "UniTE": "unite",
}
WEIGHTS = ["0", "0.001", "0.01", "0.02", "0.03", "0.05", "0.1"]


def runs():
    for row, target in TARGETS.items():
        # The old CMT20 runs replaced CMT22 with CMT20, so CMT22 is not in their fitness.
        guards = [c for c in CONTROLS if c != target]
        for w in WEIGHTS:
            args = ["--target", target, "--penalty", w, "--penalize", ",".join(guards)]
            if float(w) == 0:
                args = ["--target", target]
            if row == "CMT20":
                args += ["--comet-model", "Unbabel/wmt20-comet-da"]
            yield f"{row}_w{w}", args


if __name__ == "__main__":
    for name, args in runs():
        print(name, *map(shlex.quote, args))

"""Splitting candidate translations into GA genes and joining them back."""
from __future__ import annotations

from typing import List, Sequence

# Joins the tokens of a multi-token dictionary phrase into a single Moses gene.
PHRASE_SEP = "□"


class MosesGenes:
    """Word-level genes (Moses tokens)."""

    def __init__(self, lang: str = "en"):
        import sacremoses

        self._tok = sacremoses.MosesTokenizer(lang=lang)
        self._detok = sacremoses.MosesDetokenizer(lang=lang)

    def tokenize(self, sentence: str) -> List[str]:
        return self._tok.tokenize(sentence)

    def detokenize(self, genes: Sequence[str]) -> str:
        text = self._detok.detokenize([g for g in genes if g]).strip()
        return text.replace(PHRASE_SEP, " ")

    def phrase_gene(self, phrase: str) -> str:
        return PHRASE_SEP.join(self.tokenize(phrase))


class SentencePieceGenes:
    """Subword genes: the flores200 pieces that spBLEU itself splits text into."""

    def __init__(self):
        from sacrebleu.metrics import BLEU

        self._tokenizer = BLEU(tokenize="flores200", effective_order=True).tokenizer

    def tokenize(self, sentence: str) -> List[str]:
        return [p for p in self._tokenizer(sentence).split(" ") if p]

    def detokenize(self, genes: Sequence[str]) -> str:
        return "".join(genes).replace("▁", " ").strip()

    def phrase_gene(self, phrase: str) -> str:
        return "".join(self.tokenize(phrase))


def make_tokenizer(name: str, lang: str = "en"):
    if name == "moses":
        return MosesGenes(lang)
    if name == "sp":
        return SentencePieceGenes()
    raise ValueError(f"Unknown tokenizer: {name}")

"""The genetic algorithm. It knows nothing about metrics: fitness is a callable."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

# Some metrics crash on empty strings, so a fully deleted candidate is scored as this.
EMPTY_PLACEHOLDER = "EMPTY"


@dataclass
class Candidate:
    genes: List[str]
    text: str


@dataclass
class GAConfig:
    generations: int = 100
    mutation_rate: float = 1.0  # expected number of mutated genes per candidate
    crossover_rate: float = 0.1
    tournament_size: int = 3
    empty_slot_prob: float = 0.1  # chance to fill an empty slot chosen for mutation
    deletion_prob: float = 0.1  # chance that a mutation deletes instead of replacing
    allow_deletions: bool = True
    seed: int = 0


@dataclass
class GAResult:
    best: Candidate
    best_fitness: float
    initial_best: Candidate
    initial_best_fitness: float
    generations_run: int
    history: List[Dict] = field(default_factory=list)


def build_population(
    hypotheses: Sequence[str],
    tokenizer,
    multiplier: int,
    empty_slots: bool = True,
    max_len_factor: float = 1.3,
) -> List[Candidate]:
    """Tile the candidates into the initial population.

    With ``empty_slots`` every token is followed by an empty slot that mutation can
    fill (insertion), and chromosomes are padded to a common length so that crossover
    keeps positions aligned. Candidates keep their exact original text until changed,
    so tokenizer round trips cannot alter them.
    """
    tokenized = [tokenizer.tokenize(h) for h in hypotheses]
    if empty_slots:
        spaced = [[g for tok in toks for g in (tok, "")] for toks in tokenized]
        max_len = int(max(len(s) for s in spaced) * max_len_factor)
        chromosomes = [(s + [""] * max_len)[:max_len] for s in spaced]
    else:
        chromosomes = tokenized
    population = []
    for _ in range(multiplier):
        for genes, hyp in zip(chromosomes, hypotheses):
            population.append(Candidate(list(genes), hyp or EMPTY_PLACEHOLDER))
    return population[: len(population) & ~1]  # pairwise crossover needs an even size


def build_vocabulary(
    hypotheses: Sequence[str],
    tokenizer,
    phrases: Iterable[str] = (),
    multitoken_phrases: bool = True,
    extra_genes: Sequence[str] = (),
) -> List[str]:
    """Genes mutation can draw from: candidate tokens, dictionary phrases, a wordlist."""
    genes = {tok for h in hypotheses for tok in tokenizer.tokenize(h)}
    genes.update(phrases_to_genes(phrases, tokenizer, multitoken_phrases))
    # Sorted so that a seed reproduces a run regardless of string hash randomisation.
    return sorted(genes) + list(extra_genes)


def phrases_to_genes(phrases: Iterable[str], tokenizer, multitoken: bool = True) -> List[str]:
    genes: List[str] = []
    for phrase in phrases:
        if multitoken:
            gene = tokenizer.phrase_gene(phrase)
            if gene:
                genes.append(gene)
        else:
            genes.extend(tokenizer.tokenize(phrase))
    return genes


class GeneticAlgorithm:
    def __init__(
        self,
        fitness: Callable[[List[str]], Sequence[float]],
        tokenizer,
        vocabulary: Sequence[str],
        config: GAConfig,
        stop_at: Optional[float] = None,
    ):
        self.fitness = fitness
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.config = config
        self.stop_at = stop_at
        self.rng = np.random.default_rng(config.seed)

    def _make(self, genes: List[str]) -> Candidate:
        return Candidate(genes, self.tokenizer.detokenize(genes) or EMPTY_PLACEHOLDER)

    def mutate(self, candidate: Candidate) -> Candidate:
        cfg, rng = self.config, self.rng
        genes = candidate.genes
        if not genes or not self.vocabulary:
            return candidate
        new = list(genes)
        rate = cfg.mutation_rate / len(new)
        for i, gene in enumerate(genes):
            if rng.random() >= rate:
                continue
            # Replacing a word should be likelier than adding one into an empty slot.
            if gene == "" and rng.random() > cfg.empty_slot_prob:
                continue
            if cfg.allow_deletions and rng.random() < cfg.deletion_prob:
                new[i] = ""
            else:
                new[i] = self.vocabulary[int(rng.integers(len(self.vocabulary)))]
        return candidate if new == genes else self._make(new)

    def crossover(self, a: Candidate, b: Candidate) -> List[Candidate]:
        # A cut point strictly inside both parents needs at least 4 genes each.
        if self.rng.random() >= self.config.crossover_rate or min(len(a.genes), len(b.genes)) < 4:
            return [a, b]
        cut = int(self.rng.integers(1, min(len(a.genes), len(b.genes)) - 2))
        return [self._make(a.genes[:cut] + b.genes[cut:]), self._make(b.genes[:cut] + a.genes[cut:])]

    def select(self, population: List[Candidate], scores: np.ndarray) -> Candidate:
        contenders = self.rng.integers(0, len(population), self.config.tournament_size)
        return population[int(contenders[int(np.argmax(scores[contenders]))])]

    def run(self, population: List[Candidate]) -> GAResult:
        if not population:
            raise ValueError("Empty initial population")
        scores = np.asarray(self.fitness([c.text for c in population]), dtype=float)
        top = int(np.argmax(scores))
        initial_best, initial_fitness = population[top], float(scores[top])
        best, best_fitness = initial_best, initial_fitness
        history = [self._snapshot(0, population, scores)]
        generation = 0

        for generation in range(1, self.config.generations + 1):
            if self.stop_at is not None and best_fitness >= self.stop_at:
                generation -= 1
                break
            parents = [self.select(population, scores) for _ in population]
            children: List[Candidate] = []
            for i in range(0, len(parents), 2):
                children.extend(self.mutate(c) for c in self.crossover(parents[i], parents[i + 1]))
            population = children
            scores = np.asarray(self.fitness([c.text for c in population]), dtype=float)
            top = int(np.argmax(scores))
            if scores[top] > best_fitness:
                best, best_fitness = population[top], float(scores[top])
            history.append(self._snapshot(generation, population, scores))

        return GAResult(best, best_fitness, initial_best, initial_fitness, generation, history)

    @staticmethod
    def _snapshot(generation: int, population: List[Candidate], scores: np.ndarray) -> Dict:
        finite = scores[np.isfinite(scores)]
        top = int(np.argmax(scores))
        return {
            "generation": generation,
            "best_fitness": float(scores[top]) if math.isfinite(scores[top]) else None,
            "mean_fitness": float(finite.mean()) if finite.size else None,
            "best_text": population[top].text,
            "unscored": int(scores.size - finite.size),
        }

import numpy as np
import pytest

from ga_gemba.ga import EMPTY_PLACEHOLDER, GAConfig, GeneticAlgorithm, build_population, build_vocabulary
from ga_gemba.tokenizers import MosesGenes


@pytest.fixture(scope="module")
def tok():
    return MosesGenes()


def count_fitness(word):
    return lambda texts: [t.split().count(word) for t in texts]


def test_population_keeps_original_text_and_even_size(tok):
    hyps = ["Don't panic, it's fine.", "Second one"]
    pop = build_population(hyps, tok, multiplier=3)
    assert len(pop) == 6
    assert pop[0].text == hyps[0]
    assert len({len(c.genes) for c in pop}) == 1
    assert len(build_population(["a b", "c d", "e f"], tok, multiplier=1)) == 2


def test_padding_truncation_never_drops_tokens(tok):
    pop = build_population(["a b c d e f g h"], tok, 2)
    assert [g for g in pop[0].genes if g] == list("abcdefgh")


def test_ga_climbs_and_never_loses_the_initial_best(tok):
    hyps = ["x y z w", "z w y x"]
    pop = build_population(hyps, tok, 15)
    vocab = build_vocabulary(hyps, tok, ["good"])
    result = GeneticAlgorithm(count_fitness("good"), tok, vocab, GAConfig(generations=60, seed=1)).run(pop)
    assert result.initial_best_fitness == 0
    assert result.best_fitness >= 3
    assert result.best.text.split().count("good") == result.best_fitness


def test_zero_generations_reranks_instead_of_returning_the_first_candidate(tok):
    hyps = ["bad one", "good one", "bad two"]
    pop = build_population(hyps, tok, 2)
    result = GeneticAlgorithm(count_fitness("good"), tok, ["q"], GAConfig(generations=0)).run(pop)
    assert result.best.text == "good one"
    assert result.generations_run == 0


def test_seed_reproduces_a_run(tok):
    hyps = ["a b c", "c b a"]
    def run(seed):
        pop = build_population(hyps, tok, 5)
        return GeneticAlgorithm(count_fitness("a"), tok, build_vocabulary(hyps, tok), GAConfig(generations=20, seed=seed)).run(pop).history
    assert run(3) == run(3)
    assert run(3) != run(4)


@pytest.mark.parametrize("sentence", ["x", "x y", "x y z"])
def test_crossover_survives_short_chromosomes(tok, sentence):
    for no_slots in (False, True):
        pop = build_population([sentence, sentence], tok, 5, empty_slots=not no_slots)
        config = GAConfig(generations=10, crossover_rate=1.0, allow_deletions=not no_slots)
        GeneticAlgorithm(count_fitness("x"), tok, ["x"], config).run(pop)


def test_no_empty_mode_never_deletes(tok):
    pop = build_population(["a b c d e"], tok, 20, empty_slots=False)
    ga = GeneticAlgorithm(count_fitness("a"), tok, ["q"], GAConfig(mutation_rate=5, allow_deletions=False))
    assert all("" not in ga.mutate(c).genes for c in pop)


def test_fully_deleted_candidate_is_scored_as_placeholder(tok):
    pop = build_population(["a"], tok, 2)
    ga = GeneticAlgorithm(count_fitness("a"), tok, ["q"], GAConfig())
    assert ga._make(["", ""]).text == EMPTY_PLACEHOLDER


def test_multitoken_phrase_is_one_gene_and_detokenizes_with_spaces(tok):
    gene = tok.phrase_gene("great thing")
    assert " " not in gene
    assert tok.detokenize(["a", "", gene, "b"]) == "a great thing b"


def test_unscored_candidates_cannot_win(tok):
    def fitness(texts):
        return [float("-inf") if "bad" in t else 1.0 for t in texts]
    pop = build_population(["bad text", "ok text"], tok, 2)
    result = GeneticAlgorithm(fitness, tok, ["bad"], GAConfig(generations=5)).run(pop)
    assert "bad" not in result.best.text
    assert np.isfinite(result.best_fitness)

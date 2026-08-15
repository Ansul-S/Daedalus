"""Retrieval metrics."""

from __future__ import annotations

import math

import pytest

from daedalus.evaluation.metrics import dcg, mean, mrr, ndcg_at_k, recall_at_k

RELEVANT = {10: 2, 20: 1}


# Recall


def test_recall_counts_relevant_chunks_in_the_top_k() -> None:
    assert recall_at_k([10, 99, 20], RELEVANT, 5) == 1.0


def test_recall_is_partial_when_some_are_missing() -> None:
    assert recall_at_k([10, 99, 98], RELEVANT, 5) == 0.5


def test_recall_ignores_results_past_k() -> None:
    """The chunk is retrieved, but not in the top k."""

    assert recall_at_k([99, 98, 97, 96, 95, 10], RELEVANT, 5) == 0.0


def test_recall_of_nothing_relevant_is_undefined() -> None:
    """An unanswerable query has no recall, and scoring it zero would lie."""

    assert recall_at_k([1, 2, 3], {}, 5) is None


# MRR


def test_mrr_uses_the_first_relevant_position() -> None:
    assert mrr([99, 10, 20], RELEVANT) == 0.5


def test_mrr_is_one_when_the_first_result_is_relevant() -> None:
    assert mrr([10], RELEVANT) == 1.0


def test_mrr_is_zero_when_nothing_relevant_is_found() -> None:
    assert mrr([98, 99], RELEVANT) == 0.0


def test_mrr_of_nothing_relevant_is_undefined() -> None:
    assert mrr([1], {}) is None


# DCG and nDCG


def test_dcg_does_not_discount_the_first_position() -> None:
    assert dcg([3]) == 3.0


def test_dcg_discounts_later_positions() -> None:
    assert dcg([0, 3]) == pytest.approx(3 / math.log2(3))


def test_ndcg_is_one_for_the_ideal_ranking() -> None:
    """Essential before supporting is the best achievable order."""

    assert ndcg_at_k([10, 20], RELEVANT, 5) == pytest.approx(1.0)


def test_ndcg_punishes_ranking_supporting_above_essential() -> None:
    assert ndcg_at_k([20, 10], RELEVANT, 5) < 1.0


def test_ndcg_is_zero_when_nothing_relevant_is_retrieved() -> None:
    assert ndcg_at_k([98, 99], RELEVANT, 5) == 0.0


def test_ndcg_of_nothing_relevant_is_undefined() -> None:
    assert ndcg_at_k([1], {}, 5) is None


def test_ndcg_normalizes_across_queries_of_different_size() -> None:
    """One relevant chunk found first scores the same as six found first."""

    one = ndcg_at_k([10], {10: 2}, 5)
    many = ndcg_at_k([10, 20, 30], {10: 2, 20: 2, 30: 2}, 5)

    assert one == pytest.approx(many)


# Averaging


def test_mean_skips_undefined_scores() -> None:
    """Unanswerable queries must not depress a retrieval average."""

    assert mean([1.0, None, 0.0]) == 0.5


def test_mean_of_all_undefined_is_undefined() -> None:
    assert mean([None, None]) is None


def test_mean_of_nothing_is_undefined() -> None:
    assert mean([]) is None

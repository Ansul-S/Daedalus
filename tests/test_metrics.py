"""Tests for the ranking metrics.

Pure arithmetic, no database and no model. Every expected value is written out
as the metric's definition applied position by position, rather than copied from
a run of the implementation, so a test failing means the implementation departed
from the definition rather than from itself.
"""

from __future__ import annotations

from math import log2

import pytest

from daedalus.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    unjudged_count,
)

# Six judged chunks: two sufficient (grade 2), two contributory (grade 1), two
# unhelpful. Four are relevant at threshold 1, two at threshold 2.
POOL = {
    ("d", 1): 2,
    ("d", 2): 1,
    ("d", 3): 0,
    ("d", 4): 2,
    ("d", 5): 1,
    ("d", 6): 0,
}

# A plausible retrieval over that pool. Grades in rank order: 0, 2, 0, 1, 1.
RANKING = [("d", 3), ("d", 1), ("d", 6), ("d", 2), ("d", 5)]

# An unjudged chunk: present in the corpus, absent from the pool.
UNJUDGED = ("d", 99)


# --- precision ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("k", "threshold", "expected"),
    [
        (1, 1, 0 / 1),  # rank 1 is grade 0
        (3, 1, 1 / 3),  # only rank 2 is relevant
        (5, 1, 3 / 5),  # ranks 2, 4, 5
        (1, 2, 0 / 1),
        (3, 2, 1 / 3),  # rank 2 is the only grade 2 retrieved
        (5, 2, 1 / 5),  # ranks 4 and 5 are grade 1, so not relevant here
    ],
)
def test_precision_at_k(k: int, threshold: int, expected: float) -> None:
    assert precision_at_k(RANKING, POOL, k, threshold, "zero") == pytest.approx(
        expected
    )


def test_precision_denominator_is_positions_available_not_k() -> None:
    """A ranking shorter than k is not charged for positions holding nothing."""
    assert precision_at_k(RANKING, POOL, 10, 1, "zero") == pytest.approx(
        precision_at_k(RANKING, POOL, 5, 1, "zero")
    )


def test_precision_of_empty_ranking_is_zero() -> None:
    assert precision_at_k([], POOL, 5, 1, "zero") == 0.0


# --- recall ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("k", "threshold", "expected"),
    [
        (1, 1, 0 / 4),
        (3, 1, 1 / 4),
        (5, 1, 3 / 4),  # three of the pool's four relevant chunks
        (1, 2, 0 / 2),
        (5, 2, 1 / 2),  # one of the pool's two grade 2 chunks
    ],
)
def test_recall_at_k(k: int, threshold: int, expected: float) -> None:
    assert recall_at_k(RANKING, POOL, k, threshold, "zero") == pytest.approx(expected)


def test_recall_denominator_is_the_whole_judged_relevant_set() -> None:
    """Retrieving one relevant chunk out of four is not full recall."""
    assert recall_at_k([("d", 1)], POOL, 5, 1, "zero") == pytest.approx(1 / 4)


def test_recall_is_none_when_no_chunk_is_relevant_at_the_threshold() -> None:
    """The q41 / q45 / q51 case: judged, useful evidence, but nothing sufficient."""
    pool = {("d", 1): 1, ("d", 2): 1, ("d", 3): 0}
    assert recall_at_k([("d", 1)], pool, 5, 2, "zero") is None
    assert recall_at_k([("d", 1)], pool, 5, 1, "zero") == pytest.approx(1 / 2)


@pytest.mark.parametrize("threshold", [1, 2])
def test_recall_is_monotone_non_decreasing_in_k(threshold: int) -> None:
    values = [
        recall_at_k(RANKING, POOL, k, threshold, "zero") for k in (1, 3, 5, 7, 10)
    ]
    assert all(value is not None for value in values)
    assert values == sorted(values)  # type: ignore[type-var]


# --- reciprocal rank ---------------------------------------------------------


def test_reciprocal_rank_finds_the_first_relevant_position() -> None:
    assert reciprocal_rank(RANKING, POOL, 1, "zero") == pytest.approx(1 / 2)
    assert reciprocal_rank(RANKING, POOL, 2, "zero") == pytest.approx(1 / 2)


def test_reciprocal_rank_of_a_relevant_first_hit_is_one() -> None:
    assert reciprocal_rank([("d", 1), ("d", 3)], POOL, 2, "zero") == 1.0


def test_reciprocal_rank_distinguishes_thresholds() -> None:
    """A grade 1 first hit is relevant at threshold 1 but not at threshold 2."""
    ranking = [("d", 2), ("d", 3), ("d", 1)]
    assert reciprocal_rank(ranking, POOL, 1, "zero") == pytest.approx(1 / 1)
    assert reciprocal_rank(ranking, POOL, 2, "zero") == pytest.approx(1 / 3)


def test_reciprocal_rank_is_zero_when_nothing_relevant_is_retrieved() -> None:
    assert reciprocal_rank([("d", 3), ("d", 6)], POOL, 1, "zero") == 0.0
    assert reciprocal_rank([], POOL, 1, "zero") == 0.0


# --- NDCG --------------------------------------------------------------------


def test_ndcg_worked_by_hand() -> None:
    """The full calculation, position by position, for RANKING at k=5.

    Retrieved gains are 0, 2, 0, 1, 1 at ranks 1..5. The ideal ordering comes
    from the whole pool sorted by grade -- 2, 2, 1, 1, 0, 0 -- truncated at 5.
    Works out to roughly 0.4960.
    """
    dcg = 0 / log2(2) + 2 / log2(3) + 0 / log2(4) + 1 / log2(5) + 1 / log2(6)
    idcg = 2 / log2(2) + 2 / log2(3) + 1 / log2(4) + 1 / log2(5) + 0 / log2(6)
    assert ndcg_at_k(RANKING, POOL, 5, "zero") == pytest.approx(dcg / idcg)


def test_ndcg_idcg_uses_the_full_judged_set_not_the_retrieved_list() -> None:
    """Retrieving one relevant chunk and ranking it first is not a perfect result."""
    ranking = [("d", 1)]
    idcg = 2 / log2(2) + 2 / log2(3) + 1 / log2(4) + 1 / log2(5)
    value = ndcg_at_k(ranking, POOL, 5, "zero")
    assert value == pytest.approx(2 / idcg)
    assert value is not None and value < 1.0


def test_ndcg_ideal_ordering_is_truncated_at_k() -> None:
    """With five relevant chunks and k=2, the ideal is the best two, not all five."""
    pool = {("d", i): 2 for i in range(1, 6)} | {("d", 6): 0}
    assert ndcg_at_k([("d", 1), ("d", 2)], pool, 2, "zero") == pytest.approx(1.0)


def test_ndcg_discount_is_log2_of_position_plus_one() -> None:
    """Rank 1 is undiscounted; rank 2 is discounted by log2(3)."""
    pool = {("d", 1): 1, ("d", 2): 0}
    assert ndcg_at_k([("d", 1), ("d", 2)], pool, 2, "zero") == pytest.approx(1.0)
    assert ndcg_at_k([("d", 2), ("d", 1)], pool, 2, "zero") == pytest.approx(
        1 / log2(3)
    )


@pytest.mark.parametrize("k", [1, 3, 5, 7, 10])
def test_ndcg_of_the_ideal_ranking_is_one(k: int) -> None:
    ideal = sorted(POOL, key=lambda ref: POOL[ref], reverse=True)
    assert ndcg_at_k(ideal, POOL, k, "zero") == pytest.approx(1.0)


def test_ndcg_is_none_when_every_judged_chunk_is_grade_zero() -> None:
    pool = {("d", 1): 0, ("d", 2): 0}
    assert ndcg_at_k([("d", 1)], pool, 5, "zero") is None


def test_ndcg_uses_linear_gain_not_exponential() -> None:
    """Grade 2 is worth twice grade 1, not three times.

    Under the 2 ** g - 1 convention the gains would be 1 and 3, giving roughly
    0.7967 instead of 0.8597, so this pins the choice recorded in the module.
    """
    pool = {("d", 1): 2, ("d", 2): 1}
    ranking = [("d", 2), ("d", 1)]
    linear = (1 / log2(2) + 2 / log2(3)) / (2 / log2(2) + 1 / log2(3))
    exponential = (1 / log2(2) + 3 / log2(3)) / (3 / log2(2) + 1 / log2(3))
    value = ndcg_at_k(ranking, pool, 2, "zero")
    assert value == pytest.approx(linear)
    assert value != pytest.approx(exponential)


def test_ndcg_beyond_the_judged_pool_is_unchanged() -> None:
    """Extra k adds only grade 0 positions to both the run and the ideal."""
    assert ndcg_at_k(RANKING, POOL, 10, "zero") == pytest.approx(
        ndcg_at_k(RANKING, POOL, 5, "zero")
    )


# --- unjudged handling -------------------------------------------------------


def test_the_two_unjudged_policies_give_different_numbers() -> None:
    """An unjudged chunk at rank 1, then a grade 2 and a grade 0.

    Under "zero" the ranking is graded 0, 2, 0 and the unjudged chunk occupies
    rank 1. Under "skip" it is removed, the grade 2 becomes rank 1, and only two
    positions remain.
    """
    ranking = [UNJUDGED, ("d", 1), ("d", 3)]

    assert precision_at_k(ranking, POOL, 3, 1, "zero") == pytest.approx(1 / 3)
    assert precision_at_k(ranking, POOL, 3, 1, "skip") == pytest.approx(1 / 2)

    assert reciprocal_rank(ranking, POOL, 1, "zero") == pytest.approx(1 / 2)
    assert reciprocal_rank(ranking, POOL, 1, "skip") == pytest.approx(1 / 1)

    idcg = 2 / log2(2) + 2 / log2(3) + 1 / log2(4)
    assert ndcg_at_k(ranking, POOL, 3, "zero") == pytest.approx((2 / log2(3)) / idcg)
    assert ndcg_at_k(ranking, POOL, 3, "skip") == pytest.approx((2 / log2(2)) / idcg)


def test_recall_denominator_is_unaffected_by_the_unjudged_policy() -> None:
    """Relevance is only ever established by a judgement, so the pool is fixed."""
    ranking = [UNJUDGED, ("d", 1)]
    assert recall_at_k(ranking, POOL, 5, 1, "zero") == pytest.approx(1 / 4)
    assert recall_at_k(ranking, POOL, 5, 1, "skip") == pytest.approx(1 / 4)


@pytest.mark.parametrize("policy", ["zero", "skip"])
def test_a_wholly_unjudged_ranking_scores_zero_under_both_policies(
    policy: str,
) -> None:
    ranking = [UNJUDGED, ("d", 98)]
    assert precision_at_k(ranking, POOL, 2, 1, policy) == 0.0  # type: ignore[arg-type]
    assert reciprocal_rank(ranking, POOL, 1, policy) == 0.0  # type: ignore[arg-type]
    assert recall_at_k(ranking, POOL, 2, 1, policy) == 0.0  # type: ignore[arg-type]
    assert ndcg_at_k(ranking, POOL, 2, policy) == 0.0  # type: ignore[arg-type]


def test_unjudged_count_reports_pool_coverage() -> None:
    ranking = [UNJUDGED, ("d", 1), ("d", 98), ("d", 3)]
    assert unjudged_count(ranking, POOL, 4) == 2
    assert unjudged_count(ranking, POOL, 2) == 1
    assert unjudged_count(RANKING, POOL, 5) == 0


# --- rejected input ----------------------------------------------------------


def test_a_duplicated_chunk_in_a_ranking_is_rejected() -> None:
    ranking = [("d", 1), ("d", 3), ("d", 1)]
    with pytest.raises(ValueError, match="more than once"):
        precision_at_k(ranking, POOL, 3, 1, "zero")
    with pytest.raises(ValueError, match="more than once"):
        recall_at_k(ranking, POOL, 3, 1, "zero")
    with pytest.raises(ValueError, match="more than once"):
        reciprocal_rank(ranking, POOL, 1, "zero")
    with pytest.raises(ValueError, match="more than once"):
        ndcg_at_k(ranking, POOL, 3, "zero")
    with pytest.raises(ValueError, match="more than once"):
        unjudged_count(ranking, POOL, 3)


@pytest.mark.parametrize("k", [0, -1])
def test_a_non_positive_k_is_rejected(k: int) -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        precision_at_k(RANKING, POOL, k, 1, "zero")
    with pytest.raises(ValueError, match="k must be positive"):
        recall_at_k(RANKING, POOL, k, 1, "zero")
    with pytest.raises(ValueError, match="k must be positive"):
        ndcg_at_k(RANKING, POOL, k, "zero")
    with pytest.raises(ValueError, match="k must be positive"):
        unjudged_count(RANKING, POOL, k)


@pytest.mark.parametrize("threshold", [0, 3])
def test_a_threshold_outside_the_grading_scale_is_rejected(threshold: int) -> None:
    with pytest.raises(ValueError, match="threshold must be one of"):
        precision_at_k(RANKING, POOL, 5, threshold, "zero")
    with pytest.raises(ValueError, match="threshold must be one of"):
        recall_at_k(RANKING, POOL, 5, threshold, "zero")
    with pytest.raises(ValueError, match="threshold must be one of"):
        reciprocal_rank(RANKING, POOL, threshold, "zero")


def test_an_unknown_unjudged_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unjudged must be one of"):
        precision_at_k(RANKING, POOL, 5, 1, "assume-zero")  # type: ignore[arg-type]


def test_a_grade_outside_the_scale_is_rejected() -> None:
    pool = {("d", 1): 3}
    with pytest.raises(ValueError, match="grades must be one of"):
        precision_at_k([("d", 1)], pool, 1, 1, "zero")

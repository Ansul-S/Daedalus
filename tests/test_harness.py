"""Tests for the evaluation harness.

The scoring, aggregation and bootstrap tests are pure and need no database. The
loaders need one, and are skipped without it like the rest of the suite.
"""

from __future__ import annotations

import psycopg
import pytest

from daedalus.evaluation.harness import (
    bootstrap_ci,
    evaluate,
    load_grades,
    load_queries,
    macro_mean,
    run_retriever,
    score_ranking,
)
from daedalus.storage.queries import add_query, record_judgement

Connection = psycopg.Connection[tuple[object, ...]]

GRADES = {
    1: {("d", 1): 2, ("d", 2): 1, ("d", 3): 0},
    2: {("d", 1): 1, ("d", 2): 0},
    3: {("d", 1): 0, ("d", 2): 0},  # nothing relevant at either threshold
}
RANKINGS = {
    1: [("d", 1), ("d", 3), ("d", 2)],
    2: [("d", 2), ("d", 1)],
    3: [("d", 1), ("d", 2)],
}


# --- scoring ------------------------------------------------------------------


def test_score_ranking_covers_every_metric_and_k() -> None:
    scores = score_ranking(RANKINGS[1], GRADES[1], (1, 3), "zero")
    assert set(scores) == {
        "ndcg@1",
        "ndcg@3",
        "unjudged@1",
        "unjudged@3",
        "p@1_t1",
        "p@1_t2",
        "r@1_t1",
        "r@1_t2",
        "p@3_t1",
        "p@3_t2",
        "r@3_t1",
        "r@3_t2",
        "rr_t1",
        "rr_t2",
    }


def test_score_ranking_matches_hand_computed_values() -> None:
    """Ranking is grade 2, then 0, then 1, against a pool of 2/1/0."""
    scores = score_ranking(RANKINGS[1], GRADES[1], (1, 3), "zero")
    assert scores["p@1_t1"] == pytest.approx(1 / 1)
    assert scores["p@3_t1"] == pytest.approx(2 / 3)
    assert scores["p@3_t2"] == pytest.approx(1 / 3)
    assert scores["r@3_t1"] == pytest.approx(2 / 2)
    assert scores["r@3_t2"] == pytest.approx(1 / 1)
    assert scores["rr_t1"] == pytest.approx(1.0)
    assert scores["rr_t2"] == pytest.approx(1.0)
    assert scores["ndcg@1"] == pytest.approx(1.0)


def test_undefined_metrics_are_none_not_zero() -> None:
    """A query with nothing relevant must not contribute a zero."""
    scores = score_ranking(RANKINGS[3], GRADES[3], (1,), "zero")
    assert scores["r@1_t1"] is None
    assert scores["r@1_t2"] is None
    assert scores["ndcg@1"] is None
    assert scores["p@1_t1"] == 0.0  # precision is defined, and genuinely zero


def test_unjudged_count_is_reported_beside_the_scores() -> None:
    scores = score_ranking([("d", 1), ("d", 99)], GRADES[1], (2,), "zero")
    assert scores["unjudged@2"] == 1.0


def test_evaluate_skips_queries_with_no_judgements() -> None:
    per_query = evaluate({**RANKINGS, 99: [("d", 1)]}, GRADES, (1,), "zero")
    assert set(per_query) == {1, 2, 3}


# --- aggregation --------------------------------------------------------------


def test_macro_mean_excludes_undefined_and_reports_n() -> None:
    per_query = evaluate(RANKINGS, GRADES, (3,), "zero")
    mean, n = macro_mean(per_query, "r@3_t1")
    assert n == 2  # query 3 has no relevant chunk, so it is excluded
    assert mean == pytest.approx((1.0 + 1.0) / 2)


def test_macro_mean_is_none_when_nothing_is_defined() -> None:
    per_query = evaluate({3: RANKINGS[3]}, {3: GRADES[3]}, (1,), "zero")
    assert macro_mean(per_query, "r@1_t1") == (None, 0)


def test_macro_mean_weights_queries_equally() -> None:
    """Not a pooled count: a query with many relevant chunks does not dominate."""
    per_query = {1: {"m": 1.0}, 2: {"m": 0.0}}
    assert macro_mean(per_query, "m") == (pytest.approx(0.5), 2)


# --- bootstrap ----------------------------------------------------------------


def test_bootstrap_is_reproducible_for_a_fixed_seed() -> None:
    per_query = {i: {"m": float(i % 3) / 2} for i in range(30)}
    assert bootstrap_ci(per_query, "m", resamples=500, seed=7) == bootstrap_ci(
        per_query, "m", resamples=500, seed=7
    )


def test_bootstrap_brackets_the_point_estimate() -> None:
    per_query = {i: {"m": float(i % 3) / 2} for i in range(30)}
    low, high = bootstrap_ci(per_query, "m", resamples=2000, seed=7)  # type: ignore[misc]
    mean, _ = macro_mean(per_query, "m")
    assert mean is not None
    assert low <= mean <= high


def test_bootstrap_of_a_constant_metric_is_a_point() -> None:
    per_query = {i: {"m": 0.5} for i in range(10)}
    assert bootstrap_ci(per_query, "m", resamples=200, seed=1) == pytest.approx(
        (0.5, 0.5)
    )


def test_a_wider_spread_gives_a_wider_interval() -> None:
    tight = {i: {"m": 0.5 + (0.01 if i % 2 else -0.01)} for i in range(40)}
    loose = {i: {"m": 1.0 if i % 2 else 0.0} for i in range(40)}
    tlow, thigh = bootstrap_ci(tight, "m", resamples=2000, seed=3)  # type: ignore[misc]
    llow, lhigh = bootstrap_ci(loose, "m", resamples=2000, seed=3)  # type: ignore[misc]
    assert (thigh - tlow) < (lhigh - llow)


def test_bootstrap_ignores_undefined_queries() -> None:
    per_query = {1: {"m": 1.0}, 2: {"m": None}, 3: {"m": 1.0}}
    assert bootstrap_ci(per_query, "m", resamples=100, seed=1) == pytest.approx(
        (1.0, 1.0)
    )


def test_bootstrap_needs_at_least_two_queries() -> None:
    assert bootstrap_ci({1: {"m": 1.0}}, "m", resamples=100) is None
    assert bootstrap_ci({1: {"m": None}}, "m", resamples=100) is None


# --- loaders ------------------------------------------------------------------


def test_load_queries_and_grades_round_trip(connection: Connection) -> None:
    first = add_query(connection, "one", "authored")
    second = add_query(connection, "two", "harvested")
    assert first is not None and second is not None
    record_judgement(connection, first, "d1", 0, 2)
    record_judgement(connection, first, "d1", 1, 0)
    record_judgement(connection, second, "d1", 0, 1)

    queries = load_queries(connection)
    assert [(q[1], q[2]) for q in queries] == [
        ("one", "authored"),
        ("two", "harvested"),
    ]

    grades = load_grades(connection)
    assert grades[first] == {("d1", 0): 2, ("d1", 1): 0}
    assert grades[second] == {("d1", 0): 1}


def test_run_retriever_calls_once_per_query(connection: Connection) -> None:
    first = add_query(connection, "one", "authored")
    second = add_query(connection, "two", "harvested")
    assert first is not None and second is not None
    seen: list[str] = []

    def retriever(text: str) -> list[tuple[str, int]]:
        seen.append(text)
        return [("d1", len(text))]

    rankings = run_retriever(load_queries(connection), retriever)
    assert seen == ["one", "two"]
    assert rankings == {first: [("d1", 3)], second: [("d1", 3)]}

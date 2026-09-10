"""Tests for the duplicate safety net.

The band cuts and the draw are pure functions over similarities, so every
expectation here is worked out by hand and no test needs an embedder.
"""

from __future__ import annotations

import pytest

from daedalus import duplicates


def test_bands_run_from_the_tail_downwards() -> None:
    assert duplicates.BANDS[0] == "[0.95, 1.0]"
    assert duplicates.BANDS[-1] == "< 0.50"
    assert len(duplicates.BANDS) == 7


def test_a_similarity_lands_in_the_highest_band_it_meets() -> None:
    assert duplicates.band_for(1.0) == "[0.95, 1.0]"
    assert duplicates.band_for(0.95) == "[0.95, 1.0]"
    assert duplicates.band_for(0.9499) == "[0.90, 0.95)"
    assert duplicates.band_for(0.80) == "[0.80, 0.90)"
    assert duplicates.band_for(0.4999) == "< 0.50"
    assert duplicates.band_for(-1.0) == "< 0.50"


def test_band_counts_include_empty_bands() -> None:
    counts = duplicates.band_counts([(1, 2, 0.55), (1, 3, 0.55)])

    assert counts["[0.50, 0.60)"] == 2
    assert counts["[0.95, 1.0]"] == 0
    assert set(counts) == set(duplicates.BANDS)


def test_the_draw_is_reproducible_from_its_seed() -> None:
    pairs = [(a, a + 1, 0.55) for a in range(1, 60)]

    first = duplicates.sample_pairs(pairs, per_band=5)
    again = duplicates.sample_pairs(pairs, per_band=5)

    assert first == again


def test_a_different_seed_draws_different_pairs() -> None:
    pairs = [(a, a + 1, 0.55) for a in range(1, 60)]

    first = duplicates.sample_pairs(pairs, per_band=5, seed="one")
    other = duplicates.sample_pairs(pairs, per_band=5, seed="two")

    assert {(p[0], p[1]) for p in first} != {(p[0], p[1]) for p in other}


def test_a_short_band_is_reported_short_not_padded() -> None:
    """Borrowing from a neighbour would misdescribe what the corpus holds."""
    pairs = [(1, 2, 0.96), (3, 4, 0.55), (5, 6, 0.55), (7, 8, 0.55)]

    drawn = duplicates.sample_pairs(pairs, per_band=3)

    tail = [p for p in drawn if p[3] == "[0.95, 1.0]"]
    assert len(tail) == 1
    assert len(drawn) == 4


def test_the_draw_reports_the_band_each_pair_came_from() -> None:
    drawn = duplicates.sample_pairs([(1, 2, 0.85), (3, 4, 0.45)], per_band=5)

    assert {p[3] for p in drawn} == {"[0.80, 0.90)", "< 0.50"}


def test_the_tail_is_presented_first() -> None:
    pairs = [(1, 2, 0.20), (3, 4, 0.85), (5, 6, 0.55)]

    drawn = duplicates.sample_pairs(pairs, per_band=5)

    assert [p[2] for p in drawn] == [0.85, 0.55, 0.20]


def test_per_band_must_be_positive() -> None:
    with pytest.raises(ValueError, match="per_band must be 1 or more"):
        duplicates.sample_pairs([(1, 2, 0.5)], per_band=0)


def test_separation_counts_a_threshold_by_hand() -> None:
    labels = [
        (0.90, "duplicate"),
        (0.85, "not_duplicate"),
        (0.40, "not_duplicate"),
        (0.30, "duplicate"),
    ]

    counts = duplicates.separation(labels, 0.87)

    assert counts == {"tp": 1, "fp": 0, "tn": 2, "fn": 1}


def test_a_threshold_at_a_value_includes_that_pair() -> None:
    counts = duplicates.separation([(0.80, "duplicate")], 0.80)

    assert counts["tp"] == 1


def test_the_best_threshold_separates_cleanly_when_it_can() -> None:
    labels = [
        (0.95, "duplicate"),
        (0.92, "duplicate"),
        (0.40, "not_duplicate"),
        (0.30, "not_duplicate"),
    ]

    result = duplicates.best_threshold(labels)

    assert result is not None
    threshold, counts = result
    assert 0.40 < threshold < 0.92
    assert counts == {"tp": 2, "fp": 0, "tn": 2, "fn": 0}


def test_no_threshold_separates_interleaved_labels_cleanly() -> None:
    """The honest outcome section 11 requires be reportable."""
    labels = [
        (0.90, "duplicate"),
        (0.85, "not_duplicate"),
        (0.80, "duplicate"),
        (0.75, "not_duplicate"),
    ]

    result = duplicates.best_threshold(labels)

    assert result is not None
    _threshold, counts = result
    assert counts["fp"] + counts["fn"] > 0


def test_no_threshold_exists_when_every_pair_is_the_same_class() -> None:
    labels = [(0.9, "not_duplicate"), (0.5, "not_duplicate")]

    assert duplicates.best_threshold(labels) is None


def test_the_question_embedding_identity_is_distinct_from_production() -> None:
    """Section 11 step 1 requires the chunk embeddings be left untouched."""
    assert duplicates.QUESTION_EMBEDDING_MODEL == "bge-m3-questions"
    assert duplicates.QUESTION_EMBEDDING_MODEL != "bge-m3"

"""
Fusion is pure arithmetic over ranks, so it is tested without a database.

These are the tests that pin down *why* RRF was chosen: agreement between
retrievers outweighs confidence from one, and input scores never matter.
"""

from __future__ import annotations

import pytest

from daedalus.interfaces.retrieval import SearchHit
from daedalus.retrieval import reciprocal_rank_fusion


def hits(*chunk_ids: int) -> list[SearchHit]:
    """A ranked list where score decreases with position."""

    return [SearchHit(chunk_id=cid, score=1.0 / (i + 1)) for i, cid in enumerate(chunk_ids)]


def test_a_single_ranking_keeps_its_order() -> None:
    fused = reciprocal_rank_fusion([hits(7, 8, 9)], top_k=3)

    assert [hit.chunk_id for hit in fused] == [7, 8, 9]


def test_score_follows_the_rrf_formula() -> None:
    fused = reciprocal_rank_fusion([hits(7)], top_k=1, k=60)

    assert fused[0].score == pytest.approx(1.0 / 61)


def test_a_chunk_in_both_lists_accumulates_both_contributions() -> None:
    fused = reciprocal_rank_fusion([hits(7), hits(7)], top_k=1, k=60)

    assert fused[0].score == pytest.approx(2.0 / 61)


def test_agreement_beats_being_first_in_one_list() -> None:
    """The property hybrid retrieval is built on.

    Chunk 2 is second by one retriever and first by the other; chunk 1 is
    first by one and unseen by the other. Ranking well twice wins.
    """

    fused = reciprocal_rank_fusion([hits(1, 2), hits(2)], top_k=2)

    assert [hit.chunk_id for hit in fused] == [2, 1]


def test_input_scores_are_ignored() -> None:
    """Only position counts, which is what makes the lists combinable."""

    confident = [SearchHit(chunk_id=1, score=999.0), SearchHit(chunk_id=2, score=0.001)]
    modest = [SearchHit(chunk_id=2, score=0.002)]

    fused = reciprocal_rank_fusion([confident, modest], top_k=2)

    assert [hit.chunk_id for hit in fused] == [2, 1]


def test_ties_break_on_chunk_id() -> None:
    """Reproducibility: the evaluation harness reruns this over a frozen corpus."""

    fused = reciprocal_rank_fusion([hits(9), hits(3)], top_k=2)

    assert [hit.chunk_id for hit in fused] == [3, 9]


def test_tie_breaking_does_not_depend_on_list_order() -> None:
    forwards = reciprocal_rank_fusion([hits(9), hits(3)], top_k=2)
    backwards = reciprocal_rank_fusion([hits(3), hits(9)], top_k=2)

    assert [hit.chunk_id for hit in forwards] == [hit.chunk_id for hit in backwards]


def test_results_are_truncated_to_top_k() -> None:
    fused = reciprocal_rank_fusion([hits(1, 2, 3, 4)], top_k=2)

    assert len(fused) == 2


def test_a_chunk_appears_once_however_many_lists_found_it() -> None:
    fused = reciprocal_rank_fusion([hits(5), hits(5), hits(5)], top_k=10)

    assert [hit.chunk_id for hit in fused] == [5]


def test_no_rankings_fuse_to_nothing() -> None:
    assert reciprocal_rank_fusion([], top_k=5) == []


def test_empty_rankings_fuse_to_nothing() -> None:
    assert reciprocal_rank_fusion([[], []], top_k=5) == []


def test_asking_for_no_results_returns_none() -> None:
    assert reciprocal_rank_fusion([hits(1, 2)], top_k=0) == []


def test_a_smaller_k_sharpens_the_advantage_of_rank_one() -> None:
    """k controls how much the top of each list dominates."""

    flat = reciprocal_rank_fusion([hits(1, 2)], top_k=2, k=60)
    sharp = reciprocal_rank_fusion([hits(1, 2)], top_k=2, k=1)

    flat_gap = flat[0].score - flat[1].score
    sharp_gap = sharp[0].score - sharp[1].score

    assert sharp_gap > flat_gap

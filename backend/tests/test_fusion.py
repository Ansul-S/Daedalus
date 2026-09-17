import pytest

from app.retrieval.fusion import reciprocal_rank_fusion


def test_items_found_by_both_retrievers_rank_first() -> None:
    fused = reciprocal_rank_fusion([[10, 20, 30], [40, 30, 50]], k=60)

    assert [item for item, _ in fused] == [30, 10, 40, 20, 50]
    assert dict(fused)[30] == pytest.approx(1 / 63 + 1 / 62)
    assert dict(fused)[10] == pytest.approx(1 / 61)


def test_ties_keep_the_order_items_were_first_seen() -> None:
    fused = reciprocal_rank_fusion([["vector-hit"], ["keyword-hit"]])
    assert [item for item, _ in fused] == ["vector-hit", "keyword-hit"]


def test_a_single_ranking_keeps_its_order() -> None:
    assert [item for item, _ in reciprocal_rank_fusion([[3, 1, 2]])] == [3, 1, 2]


def test_no_rankings_give_no_results() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []

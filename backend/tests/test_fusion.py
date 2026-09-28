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


def test_a_weight_scales_what_a_ranking_contributes() -> None:
    # Item 30 is 40th in both lists, item 10 first in one. Found twice, 30 wins (2/100 against
    # 1/61); with the second list at half weight it no longer does (1.5/100).
    first = [10, *range(101, 139), 30]
    second = [*range(201, 240), 30]

    assert reciprocal_rank_fusion([first, second])[0][0] == 30

    fused = reciprocal_rank_fusion([first, second], weights=[1.0, 0.5])

    assert fused[0][0] == 10
    assert dict(fused)[30] == pytest.approx(1 / 100 + 0.5 / 100)


def test_weights_must_match_the_rankings() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[1], [2]], weights=[1.0])

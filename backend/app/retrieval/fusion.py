from collections.abc import Hashable, Sequence

# From the original RRF paper (Cormack et al., 2009); damps the weight of the top ranks.
RRF_K = 60


def reciprocal_rank_fusion[T: Hashable](
    rankings: Sequence[Sequence[T]], k: int = RRF_K
) -> list[tuple[T, float]]:
    """Merge ranked lists: each item scores sum(1 / (k + rank)) over the lists it appears in.

    Only ranks are used, so scores from different retrievers (cosine distance, ts_rank) never
    need to be made comparable. Ties keep the order in which items were first seen, so earlier
    rankings win them.
    """
    scores: dict[T, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)

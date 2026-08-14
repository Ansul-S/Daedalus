"""
Reciprocal Rank Fusion.

The problem fusion solves is that a cosine similarity and a BM25 score
have no common scale. Normalizing them onto one — min-max, z-score —
requires assuming a distribution that neither actually has, and makes the
result depend on the spread of whatever else happened to be retrieved.

RRF sidesteps the question by discarding the scores and keeping only the
ranks. A chunk's contribution from one list is ``1 / (k + rank)``, summed
across lists. Being ranked well by both retrievers beats being ranked
first by one, which is exactly the behaviour hybrid retrieval is for.

``k = 60`` comes from the original RRF paper (Cormack et al., 2009) and is
fixed in ``constants``. It flattens the curve near the top: the gap
between rank 1 and rank 2 is small relative to the gain from appearing in
both lists, so one retriever cannot dominate on confidence alone.
"""

from __future__ import annotations

from collections.abc import Sequence

from daedalus.config import constants
from daedalus.interfaces.retrieval import SearchHit

__all__ = ["reciprocal_rank_fusion"]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchHit]],
    *,
    top_k: int,
    k: int = constants.RRF_K,
) -> list[SearchHit]:
    """
    Fuse ranked result lists into one, returning the best ``top_k``.

    Input scores are ignored entirely — only the position of a hit within
    its own list matters, which is what makes lists from different
    retrievers combinable at all.
    """

    if top_k <= 0:
        return []

    scores: dict[int, float] = {}

    for ranking in rankings:
        for position, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + position)

    # Ties break on chunk id rather than on dict order. Two chunks found at
    # the same rank by the same retrievers are genuinely tied, and the
    # evaluation harness needs the same ranking on every run over a frozen
    # corpus for its metrics to be reproducible.
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))

    return [SearchHit(chunk_id=chunk_id, score=score) for chunk_id, score in ordered[:top_k]]

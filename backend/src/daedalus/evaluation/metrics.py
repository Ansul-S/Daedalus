"""
Retrieval metrics.

Three numbers, each answering a different question, because no single one
is honest on its own:

- **Recall@k** — did we fetch the material at all? The ceiling on
  everything downstream: an answer cannot be grounded in a chunk that was
  never retrieved.
- **MRR** — how far down the list was the first useful result? Cheap proxy
  for whether a reader would have to dig.
- **nDCG@k** — did the *essential* material outrank the merely supporting?
  The only one of the three that uses the two-level grades, and the only
  one that distinguishes a good ordering from a lucky one.

All of them ignore grade-0 chunks, and all of them are undefined when a
query has no relevant chunks at all. That case returns ``None`` rather than
0.0: an unanswerable query scoring zero recall would drag the mean down as
though the retriever had failed, when there was nothing to find.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

__all__ = ["dcg", "mean", "mrr", "ndcg_at_k", "recall_at_k"]


def recall_at_k(retrieved: Sequence[int], relevant: Mapping[int, int], k: int) -> float | None:
    """Share of relevant chunks appearing in the top ``k``."""

    if not relevant:
        return None

    found = sum(1 for chunk_id in retrieved[:k] if chunk_id in relevant)

    return found / len(relevant)


def mrr(retrieved: Sequence[int], relevant: Mapping[int, int]) -> float | None:
    """Reciprocal of the rank of the first relevant chunk."""

    if not relevant:
        return None

    for position, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / position

    return 0.0


def dcg(gains: Sequence[int]) -> float:
    """
    Discounted cumulative gain over a ranked list of gains.

    The ``log2(i + 1)`` discount is the standard one: position 1 is
    undiscounted, and each later position is worth progressively less.
    """

    return sum(gain / math.log2(position + 1) for position, gain in enumerate(gains, start=1))


def ndcg_at_k(retrieved: Sequence[int], relevant: Mapping[int, int], k: int) -> float | None:
    """
    Normalized DCG over the top ``k``, using the graded labels.

    Normalized against the best ranking achievable for *this* query, so a
    query with one relevant chunk and a query with six are comparable.
    """

    if not relevant:
        return None

    actual = dcg([relevant.get(chunk_id, 0) for chunk_id in retrieved[:k]])
    ideal = dcg(sorted(relevant.values(), reverse=True)[:k])

    if ideal == 0:  # pragma: no cover - grades are 1 or 2, never 0
        return None

    return actual / ideal


def mean(values: Sequence[float | None]) -> float | None:
    """
    Average, skipping queries the metric does not apply to.

    Queries scoring ``None`` are excluded rather than counted as zero, so
    the unanswerable slice cannot silently depress a retrieval number it
    was never measuring.
    """

    present = [value for value in values if value is not None]

    if not present:
        return None

    return sum(present) / len(present)

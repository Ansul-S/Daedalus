"""
The retrieval port.

Every way of finding chunks — dense, lexical, or the fusion of both —
answers the same question: given a query, which chunks, in what order.
Pinning that down as one contract is what makes the ablation in
docs/pipelines/EVALUATION_ENGINE.md possible: the harness swaps the
retriever and changes nothing else, so a difference in the numbers is
attributable to the retrieval strategy rather than to the code around it.

Hits carry chunk ids rather than chunk text. Ranking is the answer;
loading the rows is a separate concern, and the evaluation harness scores
results against character offsets without ever needing the text.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from daedalus.config import constants

__all__ = ["Retriever", "SearchHit"]


@dataclass(frozen=True)
class SearchHit:
    """
    One retrieved chunk and the score that placed it.

    Scores are higher-is-better within a single result list, but they are
    *not* comparable across implementations — a cosine similarity, a BM25
    score, and an RRF score share no scale or units. That incomparability
    is exactly why fusion consumes ranks rather than scores.
    """

    chunk_id: int
    score: float


class Retriever(ABC):
    """Finds the chunks most relevant to a query."""

    @abstractmethod
    def search(self, query: str, top_k: int = constants.DEFAULT_TOP_K) -> list[SearchHit]:
        """
        Return the best ``top_k`` chunks, most relevant first.

        Implementations must satisfy the following, which the shared
        contract test enforces:

        - At most ``top_k`` hits, ordered by descending score.
        - No duplicate chunk ids.
        - A query with nothing to match returns an empty list rather than
          raising — an unanswerable question is a normal outcome here, and
          the evaluation corpus is 15% unanswerable by design.
        - Ties break deterministically, so repeated runs over a frozen
          corpus produce identical rankings and metrics stay reproducible.
        """

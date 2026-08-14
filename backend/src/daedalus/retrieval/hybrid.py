"""
Hybrid retrieval — both indexes, fused.

The composition ADR-007 exists for: dense search covers conceptual
paraphrase, lexical search covers exact terminology, and each one's
failure mode is the other's strength.

Both arms are asked for ``HYBRID_CANDIDATES`` results rather than
``top_k``. Fusion can only reward a chunk it can see, so a chunk ranked
30th by dense search and 2nd by lexical search has to appear in the dense
list at all for the agreement between them to count. Retrieving only
``top_k`` from each arm would throw away most of the signal fusion exists
to exploit.
"""

from __future__ import annotations

import logging
import sqlite3

from daedalus.config import constants
from daedalus.interfaces.embedding import Embedder
from daedalus.interfaces.retrieval import Retriever, SearchHit
from daedalus.retrieval.dense import DenseRetriever
from daedalus.retrieval.fusion import reciprocal_rank_fusion
from daedalus.retrieval.lexical import LexicalRetriever

__all__ = ["HybridRetriever"]


logger = logging.getLogger(__name__)


class HybridRetriever(Retriever):
    """Fuses dense and lexical results with RRF."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        embedder: Embedder,
        *,
        candidates: int = constants.HYBRID_CANDIDATES,
        rrf_k: int = constants.RRF_K,
    ) -> None:
        self._dense = DenseRetriever(connection, embedder)
        self._lexical = LexicalRetriever(connection)
        self._candidates = candidates
        self._rrf_k = rrf_k

    def search(self, query: str, top_k: int = constants.DEFAULT_TOP_K) -> list[SearchHit]:
        if top_k <= 0:
            return []

        # Never fewer candidates than results asked for, or fusion would be
        # choosing the top 5 from a pool of 3.
        candidates = max(self._candidates, top_k)

        dense = self._dense.search(query, candidates)
        lexical = self._lexical.search(query, candidates)

        logger.debug(
            "Query %r: %d dense, %d lexical candidates", query, len(dense), len(lexical)
        )

        return reciprocal_rank_fusion([dense, lexical], top_k=top_k, k=self._rrf_k)

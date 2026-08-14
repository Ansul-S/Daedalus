"""
Dense retrieval — nearest neighbours in embedding space.

Answers the half of the problem keywords are bad at: a student who asks
"why does training destabilize without normalization?" shares almost no
vocabulary with a passage explaining internal covariate shift, but the two
sit close together once encoded.

Brute-force KNN, per ADR-007: sqlite-vec scans every vector rather than
building an approximate index. Linear in corpus size, which is the right
trade at the few-thousand-chunk scale here and would not be at millions.
"""

from __future__ import annotations

import logging
import sqlite3

from daedalus.config import constants
from daedalus.core.exceptions import RetrievalError
from daedalus.interfaces.embedding import Embedder
from daedalus.interfaces.retrieval import Retriever, SearchHit

__all__ = ["DenseRetriever"]


logger = logging.getLogger(__name__)


# vec0 ranks by L2 distance. For unit-length vectors — which the Embedder
# contract guarantees — L2 and cosine rank identically, and the two are
# related exactly by cos = 1 - d^2 / 2. Converting here means the port's
# "higher is better" rule holds without changing the ordering at all.
_KNN = """
SELECT chunk_id, distance
  FROM chunks_vec
 WHERE embedding MATCH ?
   AND k = ?
 ORDER BY distance
"""


class DenseRetriever(Retriever):
    """Searches ``chunks_vec`` for the nearest embeddings to a query."""

    def __init__(self, connection: sqlite3.Connection, embedder: Embedder) -> None:
        # Checked once, at construction, rather than on every search: an
        # embedder whose vectors do not fit the index can never work, and
        # the error sqlite-vec raises for a wrong-sized blob does not say so.
        if embedder.dim != constants.EMBEDDING_DIM:
            raise RetrievalError(
                f"embedder produces {embedder.dim}-dimensional vectors, "
                f"but the index is built for {constants.EMBEDDING_DIM}"
            )

        self._connection = connection
        self._embedder = embedder

    def search(self, query: str, top_k: int = constants.DEFAULT_TOP_K) -> list[SearchHit]:
        if top_k <= 0:
            return []

        vector = self._embedder.embed_query(query)

        rows = self._connection.execute(_KNN, (vector.tobytes(), top_k)).fetchall()

        return [
            SearchHit(chunk_id=row["chunk_id"], score=1.0 - (row["distance"] ** 2) / 2.0)
            for row in rows
        ]

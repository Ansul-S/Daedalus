"""
A deterministic embedder that needs no model.

Exists so the pipeline can be exercised end to end without the 2.2 GB
BGE-M3 download: tests, CI, and a first run on a machine that has never
synced the ``ml`` extra all work against this.

It lives in ``src/`` rather than ``tests/`` deliberately. The API has to be
able to boot and serve a query without the model present, and importing
test code from application code would be worse than shipping a fake.

The vectors are meaningless as semantics — nearby text does not produce
nearby vectors. What they preserve is every structural property retrieval
depends on: stable identity, unit length, and exact reproducibility.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np

from daedalus.config import constants
from daedalus.interfaces.embedding import Embedder, EmbeddingMatrix, EmbeddingVector

__all__ = ["FakeEmbedder"]


def _normalize(vector: EmbeddingVector) -> EmbeddingVector:
    """Scale a vector to unit length, leaving an all-zero vector alone."""

    norm = float(np.linalg.norm(vector))

    if norm == 0.0:
        return vector

    return np.asarray(vector / norm, dtype=np.float32)


class FakeEmbedder(Embedder):
    """
    Derives each vector from a hash of the text.

    Two properties make this usable as a test double:

    - **Reproducible across processes.** The seed comes from SHA-256, not
      Python's built-in ``hash()``, which is randomly salted per process for
      strings — vectors written to the database in one run would otherwise
      not match anything computed in the next.
    - **Pinnable.** ``overrides`` maps specific texts to specific vectors,
      so a retrieval test can lay out a known neighbourhood and assert on
      the exact fused ranking rather than on whatever the hash produced.
    """

    def __init__(
        self,
        dim: int = constants.EMBEDDING_DIM,
        *,
        overrides: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

        self._dim = dim
        self._overrides: dict[str, EmbeddingVector] = {}

        for text, values in (overrides or {}).items():
            if len(values) != dim:
                raise ValueError(f"override for {text!r} has {len(values)} values, expected {dim}")

            # Normalized on the way in, so an override cannot break the
            # unit-length guarantee the rest of the contract relies on.
            self._overrides[text] = _normalize(np.asarray(values, dtype=np.float32))

    @property
    def dim(self) -> int:
        return self._dim

    def _vector(self, text: str) -> EmbeddingVector:
        """Return the one vector this text always maps to."""

        pinned = self._overrides.get(text)

        if pinned is not None:
            return pinned

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        generator = np.random.default_rng(int.from_bytes(digest[:8], "big"))

        return _normalize(generator.standard_normal(self._dim, dtype=np.float32))

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingMatrix:
        matrix = np.empty((len(texts), self._dim), dtype=np.float32)

        for row, text in enumerate(texts):
            matrix[row] = self._vector(text)

        return matrix

    def embed_query(self, text: str) -> EmbeddingVector:
        # Identical to the document path: a query whose text exactly matches
        # a stored chunk must retrieve it at distance zero.
        return self._vector(text)

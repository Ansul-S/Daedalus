"""
The embedding port.

Retrieval quality depends heavily on the embedding model, but nothing else
in the pipeline should depend on *which* model — or on it being installed
at all. Indexing and retrieval code takes an ``Embedder``; the choice of
implementation is made once, at the edge.

The immediate payoff is testing. Fusing dense and lexical results correctly
is the hard part of hybrid retrieval, and asserting on a real model's output
means asserting on numbers nobody can predict. Against a fake with known
vectors, the expected ranking can be written down in advance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = ["Embedder", "EmbeddingMatrix", "EmbeddingVector"]


# numpy's type system cannot express array shape, so these two aliases are
# the same type at runtime. They exist to make signatures self-describing:
# a vector is 1-D ``(dim,)``, a matrix is 2-D ``(n_texts, dim)``.

EmbeddingVector = NDArray[np.float32]
EmbeddingMatrix = NDArray[np.float32]


class Embedder(ABC):
    """
    Turns text into vectors suitable for dense retrieval.

    Implementations must satisfy the following contract, which the shared
    contract test enforces:

    - ``dtype`` is ``float32`` — what ``sqlite-vec`` stores, so anything
      wider is converted on write anyway.
    - Rows are L2-normalized. With unit vectors, ranking by L2 distance and
      ranking by cosine similarity are identical, which lets the ``vec0``
      table's default metric stand in for cosine without extra work.
    - ``embed_documents`` returns one row per input, in input order.
    - Encoding is deterministic: the same text always produces the same
      vector, so a chunk's stored embedding matches what a query would
      compute for identical text.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """
        Dimensionality of the vectors produced.

        Reading this must not load a model — callers use it to validate
        against the vector table's fixed width before doing any work.
        """

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> EmbeddingMatrix:
        """
        Encode chunk text for storage.

        Returns a ``(len(texts), dim)`` matrix, including the degenerate
        ``(0, dim)`` case for empty input — callers should not have to
        special-case a document that chunked to nothing.
        """

    @abstractmethod
    def embed_query(self, text: str) -> EmbeddingVector:
        """
        Encode a search query.

        Separate from ``embed_documents`` because asymmetric models exist:
        E5 and the English BGE v1.5 models prefix queries and passages
        differently, and a symmetric implementation would silently degrade
        recall if swapped in. BGE-M3 happens to need no prefix, but that is
        a property of the adapter, not of the port.
        """

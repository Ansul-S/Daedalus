"""
The real embedder: BGE-M3 via sentence-transformers.

``sentence-transformers`` lives in the optional ``ml`` extra and pulls in
torch, and the model weights are roughly 2.2 GB on first use. Both costs are
deferred: the import happens inside ``_load``, not at module scope, so this
module can be imported — and the application started — on a machine that has
neither.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from daedalus.config import constants, settings
from daedalus.core.exceptions import EmbeddingError
from daedalus.interfaces.embedding import Embedder, EmbeddingMatrix, EmbeddingVector

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

__all__ = ["BGEEmbedder"]


logger = logging.getLogger(__name__)


class BGEEmbedder(Embedder):
    """
    Encodes text with a sentence-transformers model, loaded on first use.

    The model is held for the lifetime of the instance. Loading it costs
    seconds and gigabytes of RAM, so one instance is shared across requests
    rather than constructed per document.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        batch_size: int = constants.EMBEDDING_BATCH_SIZE,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name or settings.embedding_model
        self._batch_size = batch_size
        self._device = device
        self._model: SentenceTransformer | None = None

    @property
    def dim(self) -> int:
        # Reported from configuration rather than from the model, so this
        # stays free to call. The two are reconciled in _load(), which is
        # the only moment the real number becomes knowable.
        return constants.EMBEDDING_DIM

    def _load(self) -> SentenceTransformer:
        """Load the model once, verifying it fits the vector index."""

        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise EmbeddingError(
                "sentence-transformers is not installed. "
                "Install the ml extra with: uv sync --extra ml"
            ) from error

        logger.info("Loading embedding model %s", self._model_name)

        model = SentenceTransformer(self._model_name, device=self._device)
        actual_dim = model.get_sentence_embedding_dimension()

        # The vec0 table fixes its column width at creation time, so a model
        # of the wrong size cannot be stored. Failing here names the cause;
        # failing at insert time would surface as an opaque SQL error.
        if actual_dim != constants.EMBEDDING_DIM:
            raise EmbeddingError(
                f"{self._model_name} produces {actual_dim}-dimensional vectors, "
                f"but the index is built for {constants.EMBEDDING_DIM}. "
                f"Changing models requires a re-index."
            )

        logger.info("Embedding model ready on device %s", model.device)
        self._model = model

        return model

    def _encode(self, texts: Sequence[str]) -> EmbeddingMatrix:
        """Run the model and return normalized float32 rows."""

        model = self._load()

        # normalize_embeddings is what makes the vec0 table's L2 distance
        # rank the same way cosine similarity would.
        encoded: Any = model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return np.asarray(encoded, dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingMatrix:
        if not texts:
            # An empty encode() returns a shapeless array, and callers are
            # promised (0, dim). Short-circuiting also avoids loading the
            # model for a document that chunked to nothing.
            return np.empty((0, self.dim), dtype=np.float32)

        return self._encode(texts)

    def embed_query(self, text: str) -> EmbeddingVector:
        # BGE-M3 needs no query instruction prefix, unlike E5 and the
        # English BGE v1.5 models, so queries encode exactly like passages.
        return np.asarray(self._encode([text])[0], dtype=np.float32)

"""Text embeddings from the local Ollama server (qwen3-embedding)."""

from collections.abc import Awaitable, Callable, Sequence
from types import TracebackType
from typing import Self

import httpx

from app.core.config import Settings
from app.db.models import EMBEDDING_DIMENSIONS

# Qwen3-Embedding expects an instruction on queries only; documents are embedded as they are.
QUERY_INSTRUCTION = (
    "Given a question about machine learning, retrieve passages from study material that answer it"
)
BATCH_SIZE = 16

Progress = Callable[[int, int], Awaitable[None]]


class EmbeddingError(Exception):
    pass


def query_input(query: str) -> str:
    return f"Instruct: {QUERY_INSTRUCTION}\nQuery:{query}"


class Embedder:
    def __init__(self, settings: Settings, http: httpx.AsyncClient | None = None) -> None:
        self.model = settings.embedding_model
        # Same value on every call; a change would make Ollama reload the model.
        self.num_ctx = settings.embedding_num_ctx
        self._http = http or httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=300)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._http.aclose()

    async def embed_query(self, query: str) -> list[float]:
        [vector] = await self._embed([query_input(query)])
        return vector

    async def embed_documents(
        self, texts: Sequence[str], progress: Progress | None = None
    ) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            vectors += await self._embed(list(texts[start : start + BATCH_SIZE]))
            if progress is not None:
                await progress(len(vectors), len(texts))
        return vectors

    async def _embed(self, inputs: list[str]) -> list[list[float]]:
        payload = {
            "model": self.model,
            "input": inputs,
            # Fail instead of silently embedding a cut-off text.
            "truncate": False,
            "options": {"num_ctx": self.num_ctx},
        }
        try:
            response = await self._http.post("/api/embed", json=payload)
        except httpx.HTTPError as exc:
            raise EmbeddingError(
                f"Ollama is not reachable ({type(exc).__name__}); start it with `make ollama`"
            ) from exc
        if response.status_code != 200:
            raise EmbeddingError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:200]}"
            )
        vectors = response.json().get("embeddings") or []
        if len(vectors) != len(inputs) or any(len(v) != EMBEDDING_DIMENSIONS for v in vectors):
            expected = f"{len(inputs)} vectors of {EMBEDDING_DIMENSIONS} dimensions"
            raise EmbeddingError(f"expected {expected} from {self.model}")
        return vectors

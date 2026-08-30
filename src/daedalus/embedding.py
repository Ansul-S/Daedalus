"""Embedding generation via a local Ollama server.

Ollama exposes an HTTP endpoint, so the standard library covers this and no
HTTP dependency is needed.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

#: Embedding model served by Ollama.
DEFAULT_MODEL = "bge-m3"

#: Environment variable overriding the Ollama base URL.
OLLAMA_URL_ENV = "DAEDALUS_OLLAMA_URL"

#: Base URL used when the environment does not override it.
DEFAULT_OLLAMA_URL = "http://localhost:11434"

#: Seconds to wait for a batch. Embedding is CPU-bound and locally served, so
#: a slow response means the server is overloaded rather than unreachable.
DEFAULT_TIMEOUT = 120.0


class EmbeddingError(RuntimeError):
    """Raised when the embedding service cannot be reached or returns badly."""


def ollama_url() -> str:
    """Return the Ollama base URL, honouring the environment override."""
    return os.environ.get(OLLAMA_URL_ENV, "").strip() or DEFAULT_OLLAMA_URL


def embed_texts(
    texts: list[str],
    model: str = DEFAULT_MODEL,
    url: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[list[float]]:
    """Embed a batch of texts, returning one vector per input in order.

    Raises EmbeddingError if the server is unreachable, returns a non-JSON
    body, or returns a number of vectors that does not match the input.
    """
    if not texts:
        return []

    endpoint = f"{url or ollama_url()}/api/embed"
    payload = json.dumps({"model": model, "input": texts}).encode()
    request = urllib.request.Request(
        endpoint, data=payload, headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
    except (urllib.error.URLError, TimeoutError) as error:
        message = f"could not reach Ollama at {endpoint}: {error}"
        raise EmbeddingError(message) from error
    except json.JSONDecodeError as error:
        raise EmbeddingError(f"Ollama returned a non-JSON body: {error}") from error

    vectors = body.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise EmbeddingError(
            f"expected {len(texts)} vectors from {model}, got {_describe(vectors)}"
        )
    return [[float(value) for value in vector] for vector in vectors]


def _describe(vectors: object) -> str:
    """Describe an unexpected embeddings payload for an error message."""
    return f"{len(vectors)}" if isinstance(vectors, list) else repr(vectors)[:60]

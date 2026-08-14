"""
Shared request dependencies.

Two things every route needs and neither should construct for itself: a
database connection scoped to the request, and the embedder.

Both are overridable through FastAPI's ``dependency_overrides``, which is
what lets the tests run the real routes against a temporary database and a
model-free embedder without any branching in the application code.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from functools import lru_cache

from daedalus.config import settings
from daedalus.db import get_connection
from daedalus.embeddings import BGEEmbedder, FakeEmbedder
from daedalus.interfaces.embedding import Embedder
from daedalus.interfaces.llm import LLM
from daedalus.llm import FakeLLM, OllamaLLM

__all__ = ["get_db", "get_embedder", "get_llm"]


logger = logging.getLogger(__name__)


def get_db() -> Iterator[sqlite3.Connection]:
    """
    A connection for the duration of one request.

    Per-request rather than shared: SQLite connections carry transaction
    state, so one connection across concurrent requests would let an
    unrelated request's rollback discard another's work.
    """

    with get_connection() as connection:
        yield connection


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """
    The process-wide embedder.

    Cached because BGE-M3 costs seconds and gigabytes of RAM to load, and
    a per-request instance would pay that on every upload. Construction is
    cheap — the model itself loads on first use — so this stays safe to
    call during startup.
    """

    if settings.embedding_backend == "fake":
        logger.warning("Serving with the fake embedder: retrieval results will be meaningless")
        return FakeEmbedder()

    return BGEEmbedder()


@lru_cache(maxsize=1)
def get_llm() -> LLM:
    """
    The process-wide language model.

    Cached for the same reason as the embedder, though the cost here is a
    held HTTP configuration rather than loaded weights — Ollama keeps the
    model resident in its own process.
    """

    if settings.llm_backend == "fake":
        logger.warning("Serving with the fake LLM: answers are scripted, not generated")
        return FakeLLM()

    return OllamaLLM()

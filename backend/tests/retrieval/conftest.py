"""
Fixtures for the retrieval tests.

The corpus here is small and hand-written so that the correct answer to
every query is known in advance. Chunks 1-3 share the vocabulary of
attention and normalization; chunk 4 is deliberately off-topic.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pytest

from daedalus.db import connect, initialize_schema
from daedalus.embeddings import FakeEmbedder
from daedalus.ingestion.types import Chunk
from daedalus.interfaces.embedding import Embedder
from daedalus.storage import chunks as chunk_store
from daedalus.storage import documents

CORPUS = [
    "the dot product is scaled by the square root of d_k before the softmax",
    "softmax turns logits into a probability distribution over tokens",
    "layer normalization stabilizes training by rescaling activations",
    "convolution kernels slide across an image to extract local features",
]


@pytest.fixture
def corpus() -> list[str]:
    """The indexed texts, in the order their ids appear in ``chunk_ids``."""

    return list(CORPUS)


@pytest.fixture
def db() -> Iterator[sqlite3.Connection]:
    connection = connect(":memory:")
    initialize_schema(connection)
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture
def embedder() -> Embedder:
    return FakeEmbedder()


@pytest.fixture
def indexed(db: sqlite3.Connection, embedder: Embedder) -> sqlite3.Connection:
    """A database holding CORPUS, indexed in all three tables."""

    documents.create(
        db,
        doc_id="attention",
        filename="attention.pdf",
        source_type="arxiv",
        content_hash="hash-attention",
    )

    chunks = [
        Chunk(
            ordinal=ordinal,
            text=text,
            source_start=ordinal * 1000,
            source_end=ordinal * 1000 + len(text),
            extraction="text",
        )
        for ordinal, text in enumerate(CORPUS)
    ]

    chunk_store.replace(db, "attention", chunks, embedder.embed_documents(CORPUS))

    return db


@pytest.fixture
def chunk_ids(indexed: sqlite3.Connection) -> list[int]:
    """The database ids of CORPUS, in corpus order."""

    rows = indexed.execute("SELECT id FROM chunks ORDER BY ordinal").fetchall()

    return [row["id"] for row in rows]

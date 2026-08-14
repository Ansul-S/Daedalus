"""
The seam between the embedder and the vector index.

Everything the ``Embedder`` contract promises about dtype and normalization
only matters because ``chunks_vec`` has to be able to store the result and
rank it sensibly. This test crosses that boundary once, so a regression in
either half fails here rather than as a wrong search result later.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pytest

from daedalus.db import connect, initialize_schema
from daedalus.embeddings import FakeEmbedder

TEXTS = [
    "Attention is all you need.",
    "Retrieval-augmented generation grounds answers in sources.",
    "The mitochondrion is the powerhouse of the cell.",
]


@pytest.fixture
def indexed() -> Iterator[tuple[sqlite3.Connection, FakeEmbedder]]:
    """An in-memory database holding one vector per text in TEXTS."""

    embedder = FakeEmbedder()
    connection = connect(":memory:")

    try:
        initialize_schema(connection)

        with connection:
            for chunk_id, vector in enumerate(embedder.embed_documents(TEXTS)):
                connection.execute(
                    "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, vector.tobytes()),
                )

        yield connection, embedder
    finally:
        connection.close()


def test_a_query_retrieves_its_own_text_first(
    indexed: tuple[sqlite3.Connection, FakeEmbedder],
) -> None:
    """
    The end-to-end property dense retrieval rests on.

    It fails if the vectors are the wrong dtype, if rows lose their order
    between encoding and insertion, or if normalization is dropped.
    """

    connection, embedder = indexed
    query = embedder.embed_query(TEXTS[1])

    rows = connection.execute(
        "SELECT chunk_id, distance FROM chunks_vec "
        "WHERE embedding MATCH ? AND k = 3 ORDER BY distance",
        (query.tobytes(),),
    ).fetchall()

    # Only the winner is asserted on. The hashed vectors carry no meaning,
    # so the order of the two non-matching chunks is arbitrary.
    assert len(rows) == len(TEXTS)
    assert rows[0]["chunk_id"] == 1
    assert rows[0]["distance"] == pytest.approx(0.0, abs=1e-5)

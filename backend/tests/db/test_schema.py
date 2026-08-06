"""Tests for the database schema and connection setup."""

from __future__ import annotations

import sqlite3
import struct
from collections.abc import Iterator

import pytest

from daedalus.config import constants
from daedalus.db import SCHEMA_VERSION, connect, initialize_schema


@pytest.fixture
def db() -> Iterator[sqlite3.Connection]:
    connection = connect(":memory:")
    initialize_schema(connection)
    try:
        yield connection
    finally:
        connection.close()


def _add_document(db: sqlite3.Connection, doc_id: str = "doc-1") -> None:
    db.execute(
        """
        INSERT INTO documents
            (id, filename, source_type, content_hash, status, created_at, updated_at)
        VALUES (?, ?, 'arxiv', ?, 'completed', '2026-01-01', '2026-01-01')
        """,
        (doc_id, f"{doc_id}.pdf", f"hash-{doc_id}"),
    )


def _add_chunk(db: sqlite3.Connection, chunk_id: int, text: str, doc_id: str = "doc-1") -> None:
    db.execute(
        """
        INSERT INTO chunks (id, doc_id, ordinal, text, source_start, source_end, extraction)
        VALUES (?, ?, ?, ?, ?, ?, 'text')
        """,
        (chunk_id, doc_id, chunk_id, text, chunk_id * 100, chunk_id * 100 + len(text)),
    )


def test_schema_version_is_recorded(db: sqlite3.Connection) -> None:
    assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_initialization_is_idempotent(db: sqlite3.Connection) -> None:
    initialize_schema(db)
    initialize_schema(db)

    assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_sqlite_vec_extension_is_loaded(db: sqlite3.Connection) -> None:
    assert db.execute("SELECT vec_version()").fetchone()[0]


def test_foreign_keys_are_enforced(db: sqlite3.Connection) -> None:
    """Off by default in SQLite, so the cascade would silently not happen."""

    assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    with pytest.raises(sqlite3.IntegrityError):
        _add_chunk(db, 1, "orphaned chunk", doc_id="does-not-exist")


def test_deleting_a_document_cascades_to_chunks(db: sqlite3.Connection) -> None:
    _add_document(db)
    _add_chunk(db, 1, "some text")

    db.execute("DELETE FROM documents WHERE id = 'doc-1'")

    assert db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0


def test_invalid_status_is_rejected(db: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            """
            INSERT INTO documents
                (id, filename, source_type, content_hash, status, created_at, updated_at)
            VALUES ('d', 'd.pdf', 'arxiv', 'h', 'not-a-status', '2026-01-01', '2026-01-01')
            """
        )


def test_backwards_span_is_rejected(db: sqlite3.Connection) -> None:
    _add_document(db)

    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            """
            INSERT INTO chunks (doc_id, ordinal, text, source_start, source_end, extraction)
            VALUES ('doc-1', 0, 'text', 500, 100, 'text')
            """
        )


def test_duplicate_ordinal_within_a_document_is_rejected(db: sqlite3.Connection) -> None:
    _add_document(db)
    _add_chunk(db, 1, "first")

    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            """
            INSERT INTO chunks (doc_id, ordinal, text, source_start, source_end, extraction)
            VALUES ('doc-1', 1, 'second', 0, 5, 'text')
            """
        )


# FTS5 synchronisation


def test_fts_index_is_populated_on_insert(db: sqlite3.Connection) -> None:
    _add_document(db)
    _add_chunk(db, 1, "Self-attention lets each token attend to every other token")

    rows = db.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'attention'").fetchall()

    assert [row[0] for row in rows] == [1]


def test_fts_index_follows_deletes(db: sqlite3.Connection) -> None:
    """
    External-content FTS5 tables do not update themselves.

    Without the delete trigger, the lexical half of hybrid retrieval keeps
    returning chunks that no longer exist.
    """

    _add_document(db)
    _add_chunk(db, 1, "convolutional layers apply learned filters")

    db.execute("DELETE FROM chunks WHERE id = 1")

    rows = db.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'convolutional'"
    ).fetchall()

    assert rows == []


def test_fts_index_follows_updates(db: sqlite3.Connection) -> None:
    _add_document(db)
    _add_chunk(db, 1, "recurrent networks process sequences")

    db.execute("UPDATE chunks SET text = 'transformers process sequences' WHERE id = 1")

    assert (
        db.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'recurrent'").fetchone()[
            0
        ]
        == 0
    )
    assert (
        db.execute(
            "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'transformers'"
        ).fetchone()[0]
        == 1
    )


def test_bm25_ranking_is_available(db: sqlite3.Connection) -> None:
    _add_document(db)
    _add_chunk(db, 1, "attention attention attention mechanism")
    _add_chunk(db, 2, "a passing mention of attention in a much longer passage of other text")

    rows = db.execute(
        """
        SELECT rowid FROM chunks_fts
        WHERE chunks_fts MATCH 'attention'
        ORDER BY bm25(chunks_fts)
        """
    ).fetchall()

    assert next(row[0] for row in rows) == 1


# Vector index


def test_vector_knn_search_works(db: sqlite3.Connection) -> None:
    _add_document(db)

    vectors = {
        1: [1.0] + [0.0] * (constants.EMBEDDING_DIM - 1),
        2: [0.0, 1.0] + [0.0] * (constants.EMBEDDING_DIM - 2),
    }

    for chunk_id, vector in vectors.items():
        _add_chunk(db, chunk_id, f"chunk {chunk_id}")
        db.execute(
            "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, struct.pack(f"{constants.EMBEDDING_DIM}f", *vector)),
        )

    query = [0.9, 0.1] + [0.0] * (constants.EMBEDDING_DIM - 2)

    rows = db.execute(
        """
        SELECT chunk_id FROM chunks_vec
        WHERE embedding MATCH ? AND k = 2
        ORDER BY distance
        """,
        (struct.pack(f"{constants.EMBEDDING_DIM}f", *query),),
    ).fetchall()

    assert next(row[0] for row in rows) == 1


def test_vector_dimension_is_enforced(db: sqlite3.Connection) -> None:
    _add_document(db)
    _add_chunk(db, 1, "text")

    with pytest.raises(sqlite3.Error):
        db.execute(
            "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (1, ?)",
            (struct.pack("4f", 1.0, 2.0, 3.0, 4.0),),
        )

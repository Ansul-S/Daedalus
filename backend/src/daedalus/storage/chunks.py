"""
Writing chunks and their vectors.

Three tables have to agree for hybrid retrieval to work: ``chunks`` holds
the text, ``chunks_fts`` indexes it for BM25, and ``chunks_vec`` holds the
embedding. A chunk present in one but missing from another is the failure
mode this module exists to prevent — dense and lexical search would return
different universes of results, and the fused ranking would be quietly
wrong rather than obviously broken.

The defence is a single transaction. Either a document's chunks are fully
indexed or the database is left exactly as it was.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Sequence

import numpy as np

from daedalus.config import constants
from daedalus.core.exceptions import StorageError
from daedalus.ingestion.types import Chunk
from daedalus.interfaces.embedding import EmbeddingMatrix
from daedalus.storage.types import ChunkRecord

__all__ = ["count", "delete_for_document", "fetch", "replace"]


logger = logging.getLogger(__name__)


def _validate(chunks: Sequence[Chunk], embeddings: EmbeddingMatrix) -> None:
    """Reject a batch that cannot be stored, before touching the database."""

    if embeddings.ndim != 2:
        raise StorageError(f"embeddings must be a 2-D matrix, got {embeddings.ndim} dimension(s)")

    if len(embeddings) != len(chunks):
        raise StorageError(
            f"{len(chunks)} chunks but {len(embeddings)} embeddings — "
            f"a chunk would be stored against the wrong vector"
        )

    if embeddings.shape[1] != constants.EMBEDDING_DIM:
        raise StorageError(
            f"embeddings are {embeddings.shape[1]}-dimensional, "
            f"but the index is built for {constants.EMBEDDING_DIM}"
        )


def replace(
    connection: sqlite3.Connection,
    doc_id: str,
    chunks: Sequence[Chunk],
    embeddings: EmbeddingMatrix,
) -> int:
    """
    Store a document's chunks, discarding whatever was indexed before.

    Replacing rather than appending makes re-ingestion safe: a document can
    be chunked again with different settings without accumulating orphans,
    and a retry after a partial failure starts from a clean slate.

    Returns the number of chunks written.
    """

    _validate(chunks, embeddings)

    # float32 is what the vec0 column stores. Converting here rather than
    # trusting the caller means a float64 array cannot silently produce
    # 8-byte values that the index reads as garbage.
    vectors = np.asarray(embeddings, dtype=np.float32)

    with connection:
        # Deleting the chunk rows is enough: the FTS and vector triggers
        # clear the other two tables.
        connection.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

        for chunk, vector in zip(chunks, vectors, strict=True):
            cursor = connection.execute(
                """
                INSERT INTO chunks
                    (doc_id, ordinal, text, source_start, source_end, extraction, page)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    chunk.ordinal,
                    chunk.text,
                    chunk.source_start,
                    chunk.source_end,
                    chunk.extraction,
                    chunk.page,
                ),
            )

            # The vector table is keyed on the id SQLite just assigned, so
            # the insert has to happen one row at a time rather than as a
            # single executemany.
            connection.execute(
                "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
                (cursor.lastrowid, vector.tobytes()),
            )

    logger.info("Indexed %d chunks for document %s", len(chunks), doc_id)

    return len(chunks)


def delete_for_document(connection: sqlite3.Connection, doc_id: str) -> int:
    """Remove every chunk of a document from all three tables."""

    with connection:
        cursor = connection.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

    return cursor.rowcount


def count(connection: sqlite3.Connection, doc_id: str | None = None) -> int:
    """Count chunks, for one document or for the whole index."""

    if doc_id is None:
        row = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()
    else:
        row = connection.execute(
            "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,)
        ).fetchone()

    return int(row[0])


def fetch(connection: sqlite3.Connection, chunk_ids: Sequence[int]) -> list[ChunkRecord]:
    """
    Load chunks by id, in the order the ids were given.

    Retrieval produces a ranking of ids; the ranking is the answer, so the
    rows are reordered to match it rather than left in whatever order
    SQLite returned them.
    """

    if not chunk_ids:
        return []

    # The f-string interpolates only the placeholder marks; every id itself
    # is still bound as a parameter.
    placeholders = ",".join("?" for _ in chunk_ids)
    rows = connection.execute(
        f"SELECT * FROM chunks WHERE id IN ({placeholders})",
        tuple(chunk_ids),
    ).fetchall()

    by_id = {row["id"]: ChunkRecord.from_row(row) for row in rows}

    return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

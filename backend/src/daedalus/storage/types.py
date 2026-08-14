"""
Row types for the storage layer.

These mirror database tables exactly. They are not API schemas — nothing
here is shaped for a client, and the two should not be merged when the
request and response models arrive.

A dataclass rather than a raw ``sqlite3.Row`` so that a misspelled field is
a type error instead of a ``KeyError`` at runtime.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

__all__ = ["ChunkRecord", "DocumentRecord"]


@dataclass(frozen=True)
class DocumentRecord:
    """A row of the ``documents`` table."""

    id: str
    filename: str
    source_type: str
    content_hash: str
    status: str
    error: str | None
    n_pages: int | None
    n_chunks: int | None
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> DocumentRecord:
        return cls(
            id=row["id"],
            filename=row["filename"],
            source_type=row["source_type"],
            content_hash=row["content_hash"],
            status=row["status"],
            error=row["error"],
            n_pages=row["n_pages"],
            n_chunks=row["n_chunks"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


@dataclass(frozen=True)
class ChunkRecord:
    """
    A row of the ``chunks`` table.

    The stored counterpart of ``ingestion.types.Chunk``, with the database
    identity a chunk only acquires once written: its own ``id`` and the
    document it belongs to.
    """

    id: int
    doc_id: str
    ordinal: int
    text: str
    source_start: int
    source_end: int
    extraction: str
    page: int | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> ChunkRecord:
        return cls(
            id=row["id"],
            doc_id=row["doc_id"],
            ordinal=row["ordinal"],
            text=row["text"],
            source_start=row["source_start"],
            source_end=row["source_end"],
            extraction=row["extraction"],
            page=row["page"],
        )

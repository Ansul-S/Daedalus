"""
The document lifecycle.

A document moves ``pending -> processing -> completed | failed``, and the
API reports progress by reading that status back. Every transition goes
through this module so the timestamp is never forgotten and a transition
against a document that does not exist fails loudly rather than updating
zero rows in silence.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime

from daedalus.config import constants
from daedalus.core.exceptions import DocumentNotFoundError, DuplicateDocumentError
from daedalus.storage.types import DocumentRecord

__all__ = [
    "create",
    "get",
    "get_by_hash",
    "mark_completed",
    "mark_failed",
    "mark_processing",
    "reset_stale_processing",
]


logger = logging.getLogger(__name__)


def _now() -> str:
    """Current time as an ISO 8601 string in UTC.

    SQLite has no datetime type. Storing UTC in a fixed textual format keeps
    rows sortable as strings and free of any local-timezone ambiguity.
    """

    return datetime.now(UTC).isoformat()


# Reads


def get(connection: sqlite3.Connection, doc_id: str) -> DocumentRecord | None:
    """Return a document by id, or ``None`` if it does not exist."""

    row = connection.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()

    return None if row is None else DocumentRecord.from_row(row)


def get_by_hash(connection: sqlite3.Connection, content_hash: str) -> DocumentRecord | None:
    """
    Return the document with this content, if it has been seen before.

    The idempotency check: callers use this to answer a re-upload with the
    existing document instead of indexing the same bytes twice.
    """

    row = connection.execute(
        "SELECT * FROM documents WHERE content_hash = ?", (content_hash,)
    ).fetchone()

    return None if row is None else DocumentRecord.from_row(row)


# Writes


def create(
    connection: sqlite3.Connection,
    *,
    doc_id: str,
    filename: str,
    source_type: str,
    content_hash: str,
) -> DocumentRecord:
    """
    Record a new document as ``pending``.

    Raises ``DuplicateDocumentError`` if either the id or the content has
    been seen. Both are unique constraints, so the race between checking and
    inserting is settled by the database rather than by the caller.
    """

    timestamp = _now()

    try:
        with connection:
            connection.execute(
                """
                INSERT INTO documents
                    (id, filename, source_type, content_hash, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    filename,
                    source_type,
                    content_hash,
                    constants.STATUS_PENDING,
                    timestamp,
                    timestamp,
                ),
            )
    except sqlite3.IntegrityError as error:
        raise DuplicateDocumentError(
            f"a document with id {doc_id!r} or the same content already exists"
        ) from error

    logger.info("Registered document %s (%s)", doc_id, filename)

    created = get(connection, doc_id)

    # Unreachable in practice: the insert above committed successfully.
    if created is None:  # pragma: no cover
        raise DocumentNotFoundError(doc_id)

    return created


def _set_status(
    connection: sqlite3.Connection,
    doc_id: str,
    status: str,
    *,
    error: str | None = None,
    n_pages: int | None = None,
    n_chunks: int | None = None,
) -> None:
    """Apply a status transition, refusing to update a document that is absent."""

    with connection:
        cursor = connection.execute(
            """
            UPDATE documents
               SET status     = ?,
                   error      = ?,
                   n_pages    = COALESCE(?, n_pages),
                   n_chunks   = COALESCE(?, n_chunks),
                   updated_at = ?
             WHERE id = ?
            """,
            (status, error, n_pages, n_chunks, _now(), doc_id),
        )

        if cursor.rowcount == 0:
            raise DocumentNotFoundError(f"no document with id {doc_id!r}")


def mark_processing(connection: sqlite3.Connection, doc_id: str) -> None:
    """Move a document into ``processing`` as the background task starts."""

    _set_status(connection, doc_id, constants.STATUS_PROCESSING)


def mark_completed(
    connection: sqlite3.Connection,
    doc_id: str,
    *,
    n_pages: int | None = None,
    n_chunks: int | None = None,
) -> None:
    """
    Move a document into ``completed`` and record what was indexed.

    Clears any error from an earlier failed attempt, so a successful retry
    does not leave a stale message on a completed document.
    """

    _set_status(
        connection,
        doc_id,
        constants.STATUS_COMPLETED,
        n_pages=n_pages,
        n_chunks=n_chunks,
    )

    logger.info("Completed document %s (%s chunks)", doc_id, n_chunks)


def mark_failed(connection: sqlite3.Connection, doc_id: str, error: str) -> None:
    """Move a document into ``failed``, keeping the reason for the API to report."""

    _set_status(connection, doc_id, constants.STATUS_FAILED, error=error)

    logger.warning("Failed document %s: %s", doc_id, error)


def reset_stale_processing(connection: sqlite3.Connection) -> int:
    """
    Fail any document left mid-ingestion by a previous process.

    Background tasks do not survive a restart (ADR-008), so a document still
    marked ``processing`` at startup has no task behind it and would be
    polled forever. Returns how many were reset.
    """

    with connection:
        cursor = connection.execute(
            """
            UPDATE documents
               SET status     = ?,
                   error      = 'ingestion did not survive an application restart',
                   updated_at = ?
             WHERE status = ?
            """,
            (constants.STATUS_FAILED, _now(), constants.STATUS_PROCESSING),
        )

    if cursor.rowcount:
        logger.warning("Reset %d document(s) stuck in processing", cursor.rowcount)

    return cursor.rowcount

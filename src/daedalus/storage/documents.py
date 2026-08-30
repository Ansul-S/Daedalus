"""Writing documents and their chunks to the store.

Storing a document replaces any existing row with the same identifier. Because
the identifier is derived from content, re-ingesting unchanged material is a
no-op in effect, and re-ingesting changed material produces a new identifier
rather than overwriting the old one.
"""

from __future__ import annotations

import psycopg

from daedalus.document import Document

_INSERT_DOCUMENT = """
INSERT INTO documents (doc_id, source_path, source_format, title)
VALUES (%s, %s, %s, %s)
"""

_INSERT_CHUNK = """
INSERT INTO chunks
    (doc_id, ordinal, kind, text, heading_path, tags, locator, parent_ordinal)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""


def store_document(
    connection: psycopg.Connection[tuple[object, ...]], document: Document
) -> int:
    """Write a document and its chunks, replacing any earlier copy.

    Chunks are inserted in segment order. That order is required, not
    incidental: an output chunk carries a foreign key to the code chunk that
    produced it, so the parent must already exist.

    Returns the number of chunks written.
    """
    rows = [
        (
            document.doc_id,
            segment.ordinal,
            segment.kind.value,
            segment.text,
            list(segment.heading_path),
            list(segment.tags),
            segment.locator,
            segment.parent_ordinal,
        )
        for segment in document.segments
    ]

    with connection.cursor() as cursor:
        cursor.execute("DELETE FROM documents WHERE doc_id = %s", (document.doc_id,))
        cursor.execute(
            _INSERT_DOCUMENT,
            (
                document.doc_id,
                str(document.source_path),
                document.source_format,
                document.title,
            ),
        )
        cursor.executemany(_INSERT_CHUNK, rows)

    return len(rows)


def document_exists(
    connection: psycopg.Connection[tuple[object, ...]], doc_id: str
) -> bool:
    """Return whether a document with this identifier is already stored."""
    with connection.cursor() as cursor:
        cursor.execute("SELECT 1 FROM documents WHERE doc_id = %s", (doc_id,))
        return cursor.fetchone() is not None


def delete_document(
    connection: psycopg.Connection[tuple[object, ...]], doc_id: str
) -> None:
    """Remove a document. Its chunks and their embeddings cascade away."""
    with connection.cursor() as cursor:
        cursor.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))

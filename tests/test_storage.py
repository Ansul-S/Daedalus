"""Tests for writing documents and chunks to PostgreSQL."""

from __future__ import annotations

from pathlib import Path

import psycopg
import pytest

from daedalus.document import Document, Segment, SegmentKind
from daedalus.storage.database import (
    DATABASE_URL_ENV,
    DatabaseNotConfiguredError,
    database_url,
)
from daedalus.storage.documents import (
    delete_document,
    document_exists,
    store_document,
)

Connection = psycopg.Connection[tuple[object, ...]]


def build_document(doc_id: str = "abc123") -> Document:
    """A document exercising every chunk field, including a linked output."""
    return Document(
        doc_id=doc_id,
        source_path=Path("/corpus/notes.ipynb"),
        source_format="notebook",
        title="Section 1",
        segments=(
            Segment(
                0,
                SegmentKind.PROSE,
                "# Section 1",
                ("Section 1",),
                ("concept-check",),
                "cell:0",
            ),
            Segment(1, SegmentKind.CODE, "x = 1", ("Section 1", "1.1"), (), "cell:1"),
            Segment(
                2,
                SegmentKind.OUTPUT,
                "1",
                ("Section 1", "1.1"),
                (),
                "cell:1",
                parent_ordinal=1,
            ),
        ),
    )


def test_database_url_requires_the_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(DATABASE_URL_ENV, raising=False)
    with pytest.raises(DatabaseNotConfiguredError, match=DATABASE_URL_ENV):
        database_url()


def test_database_url_rejects_a_blank_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(DATABASE_URL_ENV, "   ")
    with pytest.raises(DatabaseNotConfiguredError):
        database_url()


def test_store_document_writes_document_and_chunks(connection: Connection) -> None:
    written = store_document(connection, build_document())

    assert written == 3
    with connection.cursor() as cur:
        cur.execute("SELECT count(*) FROM documents")
        assert cur.fetchone() == (1,)
        cur.execute("SELECT count(*) FROM chunks")
        assert cur.fetchone() == (3,)


def test_chunk_fields_round_trip(connection: Connection) -> None:
    store_document(connection, build_document())

    with connection.cursor() as cur:
        cur.execute(
            "SELECT kind, text, heading_path, tags, locator, parent_ordinal "
            "FROM chunks ORDER BY ordinal"
        )
        rows = cur.fetchall()

    assert rows[0] == (
        "prose",
        "# Section 1",
        ["Section 1"],
        ["concept-check"],
        "cell:0",
        None,
    )
    assert rows[1] == ("code", "x = 1", ["Section 1", "1.1"], [], "cell:1", None)
    assert rows[2] == ("output", "1", ["Section 1", "1.1"], [], "cell:1", 1)


def test_document_metadata_round_trips(connection: Connection) -> None:
    store_document(connection, build_document())

    with connection.cursor() as cur:
        cur.execute("SELECT doc_id, source_path, source_format, title FROM documents")
        assert cur.fetchone() == (
            "abc123",
            "/corpus/notes.ipynb",
            "notebook",
            "Section 1",
        )


def test_storing_twice_replaces_rather_than_duplicates(
    connection: Connection,
) -> None:
    store_document(connection, build_document())
    store_document(connection, build_document())

    with connection.cursor() as cur:
        cur.execute("SELECT count(*) FROM documents")
        assert cur.fetchone() == (1,)
        cur.execute("SELECT count(*) FROM chunks")
        assert cur.fetchone() == (3,)


def test_two_documents_coexist(connection: Connection) -> None:
    store_document(connection, build_document("aaa"))
    store_document(connection, build_document("bbb"))

    with connection.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks")
        assert cur.fetchone() == (6,)


def test_document_exists(connection: Connection) -> None:
    assert document_exists(connection, "abc123") is False
    store_document(connection, build_document())
    assert document_exists(connection, "abc123") is True


def test_delete_document_cascades_to_chunks(connection: Connection) -> None:
    store_document(connection, build_document())
    delete_document(connection, "abc123")

    with connection.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks")
        assert cur.fetchone() == (0,)


def test_output_chunk_without_its_parent_is_rejected(
    connection: Connection,
) -> None:
    orphan = Document(
        doc_id="bad",
        source_path=Path("/x.ipynb"),
        source_format="notebook",
        title=None,
        segments=(
            Segment(0, SegmentKind.OUTPUT, "out", (), (), "cell:0", parent_ordinal=9),
        ),
    )
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        store_document(connection, orphan)

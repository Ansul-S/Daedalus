"""
Tests for chunk indexing.

The property under test throughout is that ``chunks``, ``chunks_fts`` and
``chunks_vec`` never disagree about which chunks exist.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from daedalus.config import constants
from daedalus.core.exceptions import StorageError
from daedalus.embeddings import FakeEmbedder
from daedalus.ingestion.types import Chunk
from daedalus.interfaces.embedding import EmbeddingMatrix
from daedalus.storage import chunks as chunk_store

TEXTS = [
    "Self-attention lets each token attend to every other token.",
    "Positional encodings inject order into a permutation-invariant model.",
    "Layer normalization stabilises training of deep transformer stacks.",
]


def make_chunks(texts: list[str]) -> list[Chunk]:
    """Chunks satisfying the offset invariant a real document would have."""

    result = []
    cursor = 0

    for ordinal, text in enumerate(texts):
        result.append(
            Chunk(
                ordinal=ordinal,
                text=text,
                source_start=cursor,
                source_end=cursor + len(text),
                extraction=constants.EXTRACTION_TEXT,
                page=ordinal + 1,
            )
        )
        cursor += len(text)

    return result


def embed(texts: list[str]) -> EmbeddingMatrix:
    return FakeEmbedder().embed_documents(texts)


def _count_in(db: sqlite3.Connection, table: str) -> int:
    return int(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


# Writing


def test_chunks_land_in_all_three_tables(db: sqlite3.Connection, doc_id: str) -> None:
    written = chunk_store.replace(db, doc_id, make_chunks(TEXTS), embed(TEXTS))

    assert written == 3
    assert _count_in(db, "chunks") == 3
    assert _count_in(db, "chunks_fts") == 3
    assert _count_in(db, "chunks_vec") == 3


def test_chunk_fields_survive_the_round_trip(db: sqlite3.Connection, doc_id: str) -> None:
    chunk_store.replace(db, doc_id, make_chunks(TEXTS), embed(TEXTS))

    row = db.execute("SELECT * FROM chunks WHERE ordinal = 1").fetchone()

    assert row["text"] == TEXTS[1]
    assert row["source_start"] == len(TEXTS[0])
    assert row["source_end"] == len(TEXTS[0]) + len(TEXTS[1])
    assert row["extraction"] == constants.EXTRACTION_TEXT
    assert row["page"] == 2


def test_a_vector_is_stored_against_its_own_chunk(db: sqlite3.Connection, doc_id: str) -> None:
    """
    The off-by-one this module exists to prevent.

    Querying with the embedding of one chunk's text must return that chunk,
    not its neighbour.
    """

    embedder = FakeEmbedder()
    chunk_store.replace(db, doc_id, make_chunks(TEXTS), embedder.embed_documents(TEXTS))

    expected_id = db.execute("SELECT id FROM chunks WHERE ordinal = 2").fetchone()[0]
    nearest = db.execute(
        "SELECT chunk_id FROM chunks_vec WHERE embedding MATCH ? AND k = 1",
        (embedder.embed_query(TEXTS[2]).tobytes(),),
    ).fetchone()

    assert nearest["chunk_id"] == expected_id


def test_indexed_chunks_are_searchable_by_keyword(db: sqlite3.Connection, doc_id: str) -> None:
    chunk_store.replace(db, doc_id, make_chunks(TEXTS), embed(TEXTS))

    rows = db.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'positional'").fetchall()

    assert len(rows) == 1


def test_writing_no_chunks_is_allowed(db: sqlite3.Connection, doc_id: str) -> None:
    """A document may legitimately chunk to nothing; it must not raise."""

    assert chunk_store.replace(db, doc_id, [], embed([])) == 0


# Re-indexing


def test_reindexing_replaces_rather_than_appends(db: sqlite3.Connection, doc_id: str) -> None:
    chunk_store.replace(db, doc_id, make_chunks(TEXTS), embed(TEXTS))

    smaller = TEXTS[:1]
    chunk_store.replace(db, doc_id, make_chunks(smaller), embed(smaller))

    assert _count_in(db, "chunks") == 1
    assert _count_in(db, "chunks_fts") == 1
    assert _count_in(db, "chunks_vec") == 1


def test_reindexing_leaves_no_stale_search_hits(db: sqlite3.Connection, doc_id: str) -> None:
    chunk_store.replace(db, doc_id, make_chunks(TEXTS), embed(TEXTS))

    replacement = ["Something else entirely about convolutions."]
    chunk_store.replace(db, doc_id, make_chunks(replacement), embed(replacement))

    rows = db.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'positional'").fetchall()

    assert rows == []


def test_reindexing_does_not_touch_other_documents(db: sqlite3.Connection, doc_id: str) -> None:
    from daedalus.storage import documents

    documents.create(
        db,
        doc_id="other",
        filename="other.pdf",
        source_type="course",
        content_hash="hash-other",
    )
    chunk_store.replace(db, "other", make_chunks(TEXTS), embed(TEXTS))
    chunk_store.replace(db, doc_id, make_chunks(TEXTS), embed(TEXTS))

    chunk_store.replace(db, doc_id, [], embed([]))

    assert chunk_store.count(db, "other") == 3
    assert chunk_store.count(db, doc_id) == 0


# Deletion


def test_deleting_a_document_clears_its_vectors(db: sqlite3.Connection, doc_id: str) -> None:
    """
    ON DELETE CASCADE does not reach into a virtual table.

    Without the vector delete trigger, dense retrieval keeps returning ids
    of chunks that no longer exist, and joining them back to text yields
    nothing.
    """

    chunk_store.replace(db, doc_id, make_chunks(TEXTS), embed(TEXTS))

    with db:
        db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    assert _count_in(db, "chunks") == 0
    assert _count_in(db, "chunks_fts") == 0
    assert _count_in(db, "chunks_vec") == 0


def test_delete_for_document_reports_what_it_removed(db: sqlite3.Connection, doc_id: str) -> None:
    chunk_store.replace(db, doc_id, make_chunks(TEXTS), embed(TEXTS))

    assert chunk_store.delete_for_document(db, doc_id) == 3
    assert _count_in(db, "chunks_vec") == 0


# Validation


def test_a_mismatched_batch_is_refused(db: sqlite3.Connection, doc_id: str) -> None:
    """
    Fail before writing, not halfway through.

    Fewer embeddings than chunks would otherwise store each chunk against
    its neighbour's vector — retrieval would still work, just wrongly.
    """

    with pytest.raises(StorageError, match="chunks but"):
        chunk_store.replace(db, doc_id, make_chunks(TEXTS), embed(TEXTS[:2]))


def test_wrong_dimensionality_is_refused(db: sqlite3.Connection, doc_id: str) -> None:
    wrong = np.zeros((3, 128), dtype=np.float32)

    with pytest.raises(StorageError, match="128-dimensional"):
        chunk_store.replace(db, doc_id, make_chunks(TEXTS), wrong)


def test_a_refused_batch_writes_nothing(db: sqlite3.Connection, doc_id: str) -> None:
    chunk_store.replace(db, doc_id, make_chunks(TEXTS), embed(TEXTS))

    with pytest.raises(StorageError):
        chunk_store.replace(db, doc_id, make_chunks(TEXTS), embed(TEXTS[:1]))

    assert chunk_store.count(db, doc_id) == 3, "the previous index was destroyed by a bad batch"


def test_a_failed_write_rolls_back_completely(db: sqlite3.Connection, doc_id: str) -> None:
    """
    Atomicity across the three tables.

    A duplicate ordinal trips a constraint partway through the loop; the
    chunks written before it must not survive, or the vector and text
    indexes end up describing different documents.
    """

    chunk_store.replace(db, doc_id, make_chunks(TEXTS), embed(TEXTS))

    broken = make_chunks(TEXTS)
    broken[2] = Chunk(
        ordinal=0,  # collides with the first chunk's ordinal
        text=broken[2].text,
        source_start=broken[2].source_start,
        source_end=broken[2].source_end,
        extraction=broken[2].extraction,
    )

    with pytest.raises(sqlite3.IntegrityError):
        chunk_store.replace(db, doc_id, broken, embed(TEXTS))

    assert chunk_store.count(db, doc_id) == 3
    assert _count_in(db, "chunks_vec") == 3


def test_float64_embeddings_are_narrowed_not_corrupted(db: sqlite3.Connection, doc_id: str) -> None:
    """A float64 array would otherwise write 8-byte values into a float32 column."""

    wide = embed(TEXTS).astype(np.float64)

    assert chunk_store.replace(db, doc_id, make_chunks(TEXTS), wide) == 3
    assert _count_in(db, "chunks_vec") == 3


# Reading back


def test_fetch_returns_chunks_in_the_order_asked_for(db: sqlite3.Connection, doc_id: str) -> None:
    """Retrieval's ranking is the answer, so the rows must follow it."""

    chunk_store.replace(db, doc_id, make_chunks(TEXTS), embed(TEXTS))
    ids = [row[0] for row in db.execute("SELECT id FROM chunks ORDER BY ordinal").fetchall()]

    fetched = chunk_store.fetch(db, [ids[2], ids[0]])

    assert [record.id for record in fetched] == [ids[2], ids[0]]
    assert [record.text for record in fetched] == [TEXTS[2], TEXTS[0]]


def test_fetch_of_nothing_returns_nothing(db: sqlite3.Connection) -> None:
    assert chunk_store.fetch(db, []) == []


def test_fetch_skips_ids_that_no_longer_exist(db: sqlite3.Connection, doc_id: str) -> None:
    chunk_store.replace(db, doc_id, make_chunks(TEXTS), embed(TEXTS))
    ids = [row[0] for row in db.execute("SELECT id FROM chunks ORDER BY ordinal").fetchall()]

    fetched = chunk_store.fetch(db, [ids[0], 9999])

    assert [record.id for record in fetched] == [ids[0]]


def test_count_covers_the_whole_index_when_given_no_document(
    db: sqlite3.Connection, doc_id: str
) -> None:
    chunk_store.replace(db, doc_id, make_chunks(TEXTS), embed(TEXTS))

    assert chunk_store.count(db) == 3

"""Tests for embedding storage and backfill.

The embedder is injected, so these exercise real SQL against a real database
without contacting a model.
"""

from __future__ import annotations

from pathlib import Path

import psycopg
import pytest

from daedalus.document import Document, Segment, SegmentKind
from daedalus.storage.documents import store_document
from daedalus.storage.embeddings import (
    backfill_embeddings,
    chunks_missing_embeddings,
    embedding_input,
    store_embeddings,
)

Connection = psycopg.Connection[tuple[object, ...]]
MODEL = "test-model"


def build_document(doc_id: str = "doc1", count: int = 3) -> Document:
    return Document(
        doc_id=doc_id,
        source_path=Path("/x.ipynb"),
        source_format="notebook",
        title="T",
        segments=tuple(
            Segment(i, SegmentKind.PROSE, f"text {i}", ("A", "B"), (), f"cell:{i}")
            for i in range(count)
        ),
    )


def counting_embedder(dim: int = 4) -> tuple[object, list[list[str]]]:
    """An embedder returning fixed vectors, recording each batch it received."""
    seen: list[list[str]] = []

    def embed(texts: list[str]) -> list[list[float]]:
        seen.append(list(texts))
        return [[float(i)] * dim for i in range(len(texts))]

    return embed, seen


def test_embedding_input_prepends_the_heading_path() -> None:
    assert embedding_input(["A", "B"], "body") == "A > B\n\nbody"


def test_embedding_input_without_headings_is_the_text() -> None:
    assert embedding_input([], "body") == "body"


def test_missing_lists_every_chunk_when_nothing_embedded(
    connection: Connection,
) -> None:
    store_document(connection, build_document())

    missing = chunks_missing_embeddings(connection, MODEL, limit=10)

    assert [heading for _, heading, _ in missing] == [["A", "B"]] * 3
    assert [text for _, _, text in missing] == ["text 0", "text 1", "text 2"]


def test_missing_respects_the_limit(connection: Connection) -> None:
    store_document(connection, build_document())
    assert len(chunks_missing_embeddings(connection, MODEL, limit=2)) == 2


def test_stored_chunks_drop_out_of_missing(connection: Connection) -> None:
    store_document(connection, build_document())
    first = chunks_missing_embeddings(connection, MODEL, limit=1)[0][0]

    store_embeddings(connection, MODEL, [(first, [0.1, 0.2, 0.3, 0.4])])

    remaining = [
        chunk_id
        for chunk_id, _, _ in chunks_missing_embeddings(connection, MODEL, limit=10)
    ]
    assert first not in remaining
    assert len(remaining) == 2


def test_missing_is_per_model(connection: Connection) -> None:
    store_document(connection, build_document())
    chunk_id = chunks_missing_embeddings(connection, MODEL, limit=1)[0][0]
    store_embeddings(connection, MODEL, [(chunk_id, [1.0, 2.0])])

    other = chunks_missing_embeddings(connection, "another-model", limit=10)
    assert len(other) == 3


def test_stored_dimension_matches_the_vector(connection: Connection) -> None:
    store_document(connection, build_document(count=1))
    chunk_id = chunks_missing_embeddings(connection, MODEL, limit=1)[0][0]

    store_embeddings(connection, MODEL, [(chunk_id, [1.0, 2.0, 3.0])])

    with connection.cursor() as cur:
        cur.execute("SELECT dim, vector_dims(embedding) FROM embeddings")
        assert cur.fetchone() == (3, 3)


def test_storing_twice_replaces_the_vector(connection: Connection) -> None:
    store_document(connection, build_document(count=1))
    chunk_id = chunks_missing_embeddings(connection, MODEL, limit=1)[0][0]

    store_embeddings(connection, MODEL, [(chunk_id, [1.0, 2.0])])
    store_embeddings(connection, MODEL, [(chunk_id, [9.0, 9.0])])

    with connection.cursor() as cur:
        cur.execute("SELECT count(*), max(embedding::text) FROM embeddings")
        count, literal = cur.fetchone()
    assert count == 1
    assert literal == "[9,9]"


def test_backfill_embeds_everything(connection: Connection) -> None:
    store_document(connection, build_document(count=5))
    embed, seen = counting_embedder()

    total = backfill_embeddings(connection, embed, MODEL, batch_size=2)

    assert total == 5
    assert [len(batch) for batch in seen] == [2, 2, 1]
    assert chunks_missing_embeddings(connection, MODEL, limit=10) == []


def test_backfill_sends_the_heading_prefixed_text(connection: Connection) -> None:
    store_document(connection, build_document(count=1))
    embed, seen = counting_embedder()

    backfill_embeddings(connection, embed, MODEL, batch_size=10)

    assert seen == [["A > B\n\ntext 0"]]


def test_backfill_is_idempotent(connection: Connection) -> None:
    store_document(connection, build_document(count=3))
    embed, _ = counting_embedder()

    assert backfill_embeddings(connection, embed, MODEL, batch_size=10) == 3
    assert backfill_embeddings(connection, embed, MODEL, batch_size=10) == 0


def test_deleting_a_document_removes_its_embeddings(connection: Connection) -> None:
    store_document(connection, build_document(count=2))
    embed, _ = counting_embedder()
    backfill_embeddings(connection, embed, MODEL, batch_size=10)

    with connection.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE doc_id = 'doc1'")
        cur.execute("SELECT count(*) FROM embeddings")
        assert cur.fetchone() == (0,)


@pytest.mark.parametrize("batch_size", [1, 3, 100])
def test_backfill_batch_size_does_not_change_the_total(
    connection: Connection, batch_size: int
) -> None:
    store_document(connection, build_document(count=4))
    embed, _ = counting_embedder()

    assert backfill_embeddings(connection, embed, MODEL, batch_size=batch_size) == 4

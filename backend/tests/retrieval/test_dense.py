"""Dense retrieval over the vec0 index."""

from __future__ import annotations

import sqlite3

import pytest

from daedalus.core.exceptions import RetrievalError
from daedalus.embeddings import FakeEmbedder
from daedalus.interfaces.embedding import Embedder
from daedalus.retrieval import DenseRetriever


def test_a_query_retrieves_its_own_text_first(
    indexed: sqlite3.Connection, embedder: Embedder, chunk_ids: list[int], corpus: list[str]
) -> None:
    hits = DenseRetriever(indexed, embedder).search(corpus[2])

    assert hits[0].chunk_id == chunk_ids[2]


def test_an_exact_match_scores_one(
    indexed: sqlite3.Connection, embedder: Embedder, corpus: list[str]
) -> None:
    """Distance zero converts to cosine similarity 1."""

    hits = DenseRetriever(indexed, embedder).search(corpus[2])

    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_scores_descend(indexed: sqlite3.Connection, embedder: Embedder, corpus: list[str]) -> None:
    hits = DenseRetriever(indexed, embedder).search(corpus[0], top_k=4)

    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


def test_results_are_limited_to_top_k(
    indexed: sqlite3.Connection, embedder: Embedder, corpus: list[str]
) -> None:
    hits = DenseRetriever(indexed, embedder).search(corpus[0], top_k=2)

    assert len(hits) == 2


def test_every_chunk_is_reachable(
    indexed: sqlite3.Connection, embedder: Embedder, chunk_ids: list[int], corpus: list[str]
) -> None:
    """Brute-force KNN scans everything, so a large enough k returns all of it."""

    hits = DenseRetriever(indexed, embedder).search(corpus[0], top_k=len(corpus))

    assert {hit.chunk_id for hit in hits} == set(chunk_ids)


def test_no_chunk_is_returned_twice(
    indexed: sqlite3.Connection, embedder: Embedder, corpus: list[str]
) -> None:
    hits = DenseRetriever(indexed, embedder).search(corpus[0], top_k=len(corpus))

    assert len({hit.chunk_id for hit in hits}) == len(hits)


def test_asking_for_no_results_returns_none(
    indexed: sqlite3.Connection, embedder: Embedder, corpus: list[str]
) -> None:
    assert DenseRetriever(indexed, embedder).search(corpus[0], top_k=0) == []


def test_an_empty_index_returns_nothing(
    db: sqlite3.Connection, embedder: Embedder
) -> None:
    """An empty corpus is a normal state, not an error."""

    assert DenseRetriever(db, embedder).search("anything") == []


def test_an_embedder_that_does_not_fit_the_index_is_refused(
    db: sqlite3.Connection,
) -> None:
    """Caught at construction, before a confusing blob-size error from sqlite-vec."""

    with pytest.raises(RetrievalError, match="384"):
        DenseRetriever(db, FakeEmbedder(dim=384))


def test_deleted_chunks_stop_matching(
    indexed: sqlite3.Connection, embedder: Embedder, chunk_ids: list[int], corpus: list[str]
) -> None:
    """Guards the chunks_vec delete trigger from the storage layer."""

    indexed.execute("DELETE FROM chunks WHERE id = ?", (chunk_ids[2],))
    indexed.commit()

    hits = DenseRetriever(indexed, embedder).search(corpus[2], top_k=len(corpus))

    assert chunk_ids[2] not in {hit.chunk_id for hit in hits}

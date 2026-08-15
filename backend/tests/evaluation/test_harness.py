"""Indexing the frozen corpus and scoring retrievers against it."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pytest

from daedalus.db import connect, initialize_schema
from daedalus.embeddings import FakeEmbedder
from daedalus.evaluation.corpus import FrozenDocument
from daedalus.evaluation.dataset import EvalQuery
from daedalus.evaluation.harness import evaluate, index_frozen
from daedalus.ingestion.types import Segment
from daedalus.interfaces.embedding import Embedder
from daedalus.retrieval import LexicalRetriever

TEXT = (
    "Scaled dot-product attention divides by the square root of d_k. " * 12
    + "Layer normalization rescales activations across the feature dimension. " * 12
    + "Convolution kernels extract local features from an image. " * 12
)


@pytest.fixture
def embedder() -> Embedder:
    return FakeEmbedder()


@pytest.fixture
def db() -> Iterator[sqlite3.Connection]:
    connection = connect(":memory:")
    initialize_schema(connection)
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture
def documents() -> list[FrozenDocument]:
    return [
        FrozenDocument(
            doc_id="paper",
            filename="paper.pdf",
            source_type="arxiv",
            text=TEXT,
            segments=(Segment(start=0, end=len(TEXT), extraction="text", page=1),),
            n_pages=1,
            text_sha256="hash-paper",
        )
    ]


def make_query(char_start: int, char_end: int, **overrides: object) -> EvalQuery:
    record: dict[str, object] = {
        "id": "ret-0001",
        "query": "scaled dot-product attention",
        "query_type": "exact_term",
        "source_type": "arxiv",
        "split": "dev",
        "relevant_spans": [
            {
                "doc_id": "paper",
                "char_start": char_start,
                "char_end": char_end,
                "quote": TEXT[char_start:char_end][:20],
                "grade": 2,
            }
        ],
    }
    record.update(overrides)

    return EvalQuery.model_validate(record)


# Indexing


def test_the_frozen_corpus_is_indexed(
    db: sqlite3.Connection, documents: list[FrozenDocument], embedder: Embedder
) -> None:
    corpus = index_frozen(db, documents, embedder)

    assert corpus.n_chunks > 1


def test_chunk_ids_line_up_with_chunks(
    db: sqlite3.Connection, documents: list[FrozenDocument], embedder: Embedder
) -> None:
    """Resolution depends on these two staying parallel."""

    corpus = index_frozen(db, documents, embedder)

    assert len(corpus.chunks_by_doc["paper"]) == len(corpus.ids_by_doc["paper"])


def test_chunk_size_is_configurable(
    db: sqlite3.Connection, documents: list[FrozenDocument], embedder: Embedder
) -> None:
    """Chunking is not frozen, because chunk size is what gets measured."""

    small = index_frozen(db, documents, embedder, chunk_size=300, overlap=50)

    assert small.n_chunks > 3


# Resolution


def test_a_label_resolves_to_a_chunk(
    db: sqlite3.Connection, documents: list[FrozenDocument], embedder: Embedder
) -> None:
    corpus = index_frozen(db, documents, embedder)

    relevant = corpus.relevant_for(make_query(50, 300))

    assert relevant
    assert set(relevant.values()) == {2}


def test_a_label_on_an_unindexed_document_resolves_to_nothing(
    db: sqlite3.Connection, documents: list[FrozenDocument], embedder: Embedder
) -> None:
    corpus = index_frozen(db, documents, embedder)

    query = make_query(50, 300)
    query.relevant_spans[0].doc_id = "elsewhere"

    assert corpus.relevant_for(query) == {}


# Scoring


def test_every_query_is_scored(
    db: sqlite3.Connection, documents: list[FrozenDocument], embedder: Embedder
) -> None:
    corpus = index_frozen(db, documents, embedder)
    queries = [make_query(50, 300), make_query(50, 300, id="ret-0002")]

    result = evaluate(LexicalRetriever(db), corpus, queries, name="lexical")

    assert len(result.scores) == 2


def test_a_matching_query_scores_above_zero(
    db: sqlite3.Connection, documents: list[FrozenDocument], embedder: Embedder
) -> None:
    corpus = index_frozen(db, documents, embedder)

    result = evaluate(LexicalRetriever(db), corpus, [make_query(0, 300)], name="lexical")

    assert result.summary()["recall"] is not None
    assert result.summary()["recall"] > 0


def test_unanswerable_queries_are_excluded_from_the_average(
    db: sqlite3.Connection, documents: list[FrozenDocument], embedder: Embedder
) -> None:
    """Scoring them zero would look like a retrieval failure that never happened."""

    corpus = index_frozen(db, documents, embedder)
    unanswerable = EvalQuery.model_validate(
        {
            "id": "ret-0099",
            "query": "photosynthesis in vascular plants",
            "query_type": "unanswerable",
            "source_type": "arxiv",
            "answerable": False,
            "split": "dev",
            "relevant_spans": [],
        }
    )

    result = evaluate(LexicalRetriever(db), corpus, [unanswerable], name="lexical")

    assert result.summary()["recall"] is None


def test_results_can_be_sliced(
    db: sqlite3.Connection, documents: list[FrozenDocument], embedder: Embedder
) -> None:
    corpus = index_frozen(db, documents, embedder)
    queries = [
        make_query(0, 300),
        make_query(0, 300, id="ret-0002", query_type="conceptual"),
    ]

    result = evaluate(LexicalRetriever(db), corpus, queries, name="lexical")

    assert set(result.sliced_by("query_type")) == {"exact_term", "conceptual"}


def test_unanswerable_queries_still_returning_chunks_are_counted(
    db: sqlite3.Connection, documents: list[FrozenDocument], embedder: Embedder
) -> None:
    corpus = index_frozen(db, documents, embedder)
    unanswerable = EvalQuery.model_validate(
        {
            "id": "ret-0099",
            "query": "attention",
            "query_type": "unanswerable",
            "source_type": "arxiv",
            "answerable": False,
            "split": "dev",
            "relevant_spans": [],
        }
    )

    result = evaluate(LexicalRetriever(db), corpus, [unanswerable], name="lexical")

    assert result.unanswerable_returning_results == 1

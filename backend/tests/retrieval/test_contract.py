"""
The contract every Retriever must satisfy.

Run against all three implementations. The evaluation harness swaps
retrievers and compares the numbers, which is only meaningful if they
agree on the shape of an answer — a retriever that returned duplicates or
ranked ascending would produce metrics that look like a quality
difference rather than a bug.
"""

from __future__ import annotations

import sqlite3

import pytest

from daedalus.interfaces.embedding import Embedder
from daedalus.interfaces.retrieval import Retriever
from daedalus.retrieval import DenseRetriever, HybridRetriever, LexicalRetriever

IMPLEMENTATIONS = ["dense", "lexical", "hybrid"]


@pytest.fixture(params=IMPLEMENTATIONS)
def retriever(
    request: pytest.FixtureRequest, indexed: sqlite3.Connection, embedder: Embedder
) -> Retriever:
    if request.param == "dense":
        return DenseRetriever(indexed, embedder)

    if request.param == "lexical":
        return LexicalRetriever(indexed)

    return HybridRetriever(indexed, embedder)


def test_returns_no_more_than_top_k(retriever: Retriever, corpus: list[str]) -> None:
    assert len(retriever.search(corpus[0], top_k=2)) <= 2


def test_scores_are_descending(retriever: Retriever, corpus: list[str]) -> None:
    hits = retriever.search(corpus[0], top_k=4)

    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


def test_no_chunk_appears_twice(retriever: Retriever, corpus: list[str]) -> None:
    hits = retriever.search(corpus[0], top_k=4)

    assert len({hit.chunk_id for hit in hits}) == len(hits)


def test_asking_for_no_results_returns_none(retriever: Retriever, corpus: list[str]) -> None:
    assert retriever.search(corpus[0], top_k=0) == []


def test_a_query_matching_nothing_is_not_an_error(retriever: Retriever) -> None:
    """15% of the evaluation corpus is unanswerable by design."""

    retriever.search("photosynthesis in vascular plants")


def test_a_hostile_query_is_not_an_error(retriever: Retriever) -> None:
    retriever.search('C++ "quoted" NEAR(a b)')


def test_ranking_is_reproducible(retriever: Retriever, corpus: list[str]) -> None:
    """Frozen corpus plus frozen query must give the same ranking every run."""

    first = retriever.search(corpus[0], top_k=4)
    second = retriever.search(corpus[0], top_k=4)

    assert [hit.chunk_id for hit in first] == [hit.chunk_id for hit in second]


def test_retriever_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        Retriever()  # type: ignore[abstract]

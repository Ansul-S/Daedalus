"""
Hybrid retrieval.

The central test here is ``test_hybrid_rescues_an_exact_term_dense_search_misses``.
EVALUATION_ENGINE.md sets the bar explicitly: if hybrid does not beat
dense-only on the exact-term slice, RRF is not earning its complexity. The
neighbourhood below is pinned so that claim is checked as arithmetic
rather than measured against a model's guesses.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from daedalus.config import constants
from daedalus.embeddings import FakeEmbedder
from daedalus.ingestion.types import Chunk
from daedalus.retrieval import DenseRetriever, HybridRetriever, LexicalRetriever
from daedalus.storage import chunks as chunk_store
from daedalus.storage import documents

QUERY = "rmsnorm"

# The chunk that answers the query. Dense search ranks it second.
TARGET = "rmsnorm normalizes activations over the last dimension"

# Semantically nearest to the query but lexically irrelevant: it does not
# contain the term at all. Dense search ranks it first.
DECOY = "a technique for stabilizing very deep networks during training"

FILLERS = [
    "convolution kernels slide across an image",
    "tokenizers split text into subword units",
]


def _unit(*components: tuple[int, float]) -> list[float]:
    """A unit vector with the given weights on the given axes."""

    vector = np.zeros(constants.EMBEDDING_DIM, dtype=np.float32)

    for axis, weight in components:
        vector[axis] = weight

    return [float(value) for value in vector / np.linalg.norm(vector)]


@pytest.fixture
def pinned() -> FakeEmbedder:
    """An embedder laying out a known neighbourhood around the query.

    Cosine to the query: DECOY 1.0, TARGET 0.9, fillers 0.0. So dense
    search returns DECOY first and TARGET second, by construction.
    """

    return FakeEmbedder(
        overrides={
            QUERY: _unit((0, 1.0)),
            DECOY: _unit((0, 1.0)),
            TARGET: _unit((0, 0.9), (1, np.sqrt(1 - 0.81))),
            FILLERS[0]: _unit((2, 1.0)),
            FILLERS[1]: _unit((3, 1.0)),
        }
    )


@pytest.fixture
def neighbourhood(db: sqlite3.Connection, pinned: FakeEmbedder) -> dict[str, int]:
    """Index the pinned texts, returning each one's chunk id."""

    documents.create(
        db,
        doc_id="norms",
        filename="norms.pdf",
        source_type="arxiv",
        content_hash="hash-norms",
    )

    texts = [DECOY, TARGET, *FILLERS]
    chunks = [
        Chunk(
            ordinal=ordinal,
            text=text,
            source_start=ordinal * 1000,
            source_end=ordinal * 1000 + len(text),
            extraction="text",
        )
        for ordinal, text in enumerate(texts)
    ]

    chunk_store.replace(db, "norms", chunks, pinned.embed_documents(texts))

    rows = db.execute("SELECT id, text FROM chunks").fetchall()

    return {row["text"]: row["id"] for row in rows}


def test_dense_search_alone_prefers_the_decoy(
    db: sqlite3.Connection, pinned: FakeEmbedder, neighbourhood: dict[str, int]
) -> None:
    """The failure hybrid retrieval exists to correct."""

    hits = DenseRetriever(db, pinned).search(QUERY)

    assert hits[0].chunk_id == neighbourhood[DECOY]


def test_lexical_search_alone_finds_only_the_target(
    db: sqlite3.Connection, neighbourhood: dict[str, int]
) -> None:
    hits = LexicalRetriever(db).search(QUERY)

    assert [hit.chunk_id for hit in hits] == [neighbourhood[TARGET]]


def test_hybrid_rescues_an_exact_term_dense_search_misses(
    db: sqlite3.Connection, pinned: FakeEmbedder, neighbourhood: dict[str, int]
) -> None:
    """Ranked 2nd and 1st beats ranked 1st and nowhere.

    TARGET scores 1/62 + 1/61; DECOY scores 1/61 alone.
    """

    hits = HybridRetriever(db, pinned).search(QUERY)

    assert hits[0].chunk_id == neighbourhood[TARGET]
    assert hits[1].chunk_id == neighbourhood[DECOY]


def test_the_decoy_is_not_discarded_only_demoted(
    db: sqlite3.Connection, pinned: FakeEmbedder, neighbourhood: dict[str, int]
) -> None:
    """Fusion reorders the union; it does not filter one retriever out."""

    hits = HybridRetriever(db, pinned).search(QUERY, top_k=4)

    assert neighbourhood[DECOY] in {hit.chunk_id for hit in hits}


def test_results_are_limited_to_top_k(
    db: sqlite3.Connection, pinned: FakeEmbedder, neighbourhood: dict[str, int]
) -> None:
    assert len(HybridRetriever(db, pinned).search(QUERY, top_k=2)) == 2


def test_candidates_are_never_fewer_than_the_results_asked_for(
    db: sqlite3.Connection, pinned: FakeEmbedder, neighbourhood: dict[str, int]
) -> None:
    """A small candidate pool must not silently cap the result count."""

    retriever = HybridRetriever(db, pinned, candidates=1)

    assert len(retriever.search(QUERY, top_k=4)) == 4


def test_scores_descend(
    db: sqlite3.Connection, pinned: FakeEmbedder, neighbourhood: dict[str, int]
) -> None:
    hits = HybridRetriever(db, pinned).search(QUERY, top_k=4)

    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


def test_no_chunk_is_returned_twice(
    db: sqlite3.Connection, pinned: FakeEmbedder, neighbourhood: dict[str, int]
) -> None:
    """A chunk both arms found must be merged, not listed twice."""

    hits = HybridRetriever(db, pinned).search(QUERY, top_k=4)

    assert len({hit.chunk_id for hit in hits}) == len(hits)


def test_a_query_with_no_searchable_terms_still_returns_dense_results(
    db: sqlite3.Connection, pinned: FakeEmbedder, neighbourhood: dict[str, int]
) -> None:
    """One arm returning nothing degrades the ranking, it does not break it."""

    assert HybridRetriever(db, pinned).search("?!*") != []


def test_an_empty_index_returns_nothing(db: sqlite3.Connection, pinned: FakeEmbedder) -> None:
    assert HybridRetriever(db, pinned).search(QUERY) == []


def test_asking_for_no_results_returns_none(
    db: sqlite3.Connection, pinned: FakeEmbedder, neighbourhood: dict[str, int]
) -> None:
    assert HybridRetriever(db, pinned).search(QUERY, top_k=0) == []

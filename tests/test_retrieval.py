"""Tests for retrieval and candidate pooling.

Real SQL against a real database; the embedder is a stub, so no model is used.
"""

from __future__ import annotations

from pathlib import Path

import psycopg

from daedalus.document import Document, Segment, SegmentKind
from daedalus.retrieval.search import (
    LEXICAL,
    RANDOM,
    VECTOR,
    lexical_search,
    pool_candidates,
    random_chunks,
    vector_search,
)
from daedalus.storage.documents import store_document
from daedalus.storage.embeddings import store_embeddings

Connection = psycopg.Connection[tuple[object, ...]]
MODEL = "test-model"

TEXTS = [
    "the reader model predicts a start and end span over the passage",
    "vector databases store embeddings and support similarity search",
    "bagging reduces variance by averaging many decision trees",
    "gradient descent updates weights in the direction of steepest descent",
    "tokenisation splits text into subword units before encoding",
]


def seed(connection: Connection, vectors: dict[int, list[float]] | None = None) -> None:
    """Store five chunks, optionally with embeddings by position."""
    document = Document(
        doc_id="d1",
        source_path=Path("/x.ipynb"),
        source_format="notebook",
        title="T",
        segments=tuple(
            Segment(i, SegmentKind.PROSE, text, ("A",), (), f"cell:{i}")
            for i, text in enumerate(TEXTS)
        ),
    )
    store_document(connection, document)

    if vectors:
        with connection.cursor() as cursor:
            cursor.execute("SELECT id, ordinal FROM chunks ORDER BY ordinal")
            ids = {int(ordinal): int(cid) for cid, ordinal in cursor.fetchall()}
        store_embeddings(connection, MODEL, [(ids[o], v) for o, v in vectors.items()])


def test_lexical_search_finds_the_matching_chunk(connection: Connection) -> None:
    seed(connection)

    results = lexical_search(connection, "reader span passage", limit=5)

    assert results
    assert results[0].text == TEXTS[0]


def test_lexical_search_stems_words(connection: Connection) -> None:
    seed(connection)

    # "predicts" in the text, "predict" in the query
    results = lexical_search(connection, "predict", limit=5)

    assert [r.ordinal for r in results] == [0]


def test_lexical_search_returns_nothing_for_absent_terms(
    connection: Connection,
) -> None:
    seed(connection)
    assert lexical_search(connection, "kubernetes helm chart", limit=5) == []


def test_lexical_search_tolerates_punctuation(connection: Connection) -> None:
    seed(connection)
    # to_tsquery would raise on this; websearch_to_tsquery must not
    assert lexical_search(connection, "what is a reader? (span!)", limit=5)


def test_lexical_search_respects_the_limit(connection: Connection) -> None:
    seed(connection)
    assert len(lexical_search(connection, "the", limit=2)) <= 2


def test_vector_search_orders_by_cosine_distance(connection: Connection) -> None:
    seed(connection, vectors={0: [1.0, 0.0], 1: [0.0, 1.0], 2: [0.9, 0.1]})

    results = vector_search(connection, [1.0, 0.0], MODEL, limit=3)

    assert [r.ordinal for r in results] == [0, 2, 1]


def test_vector_search_ignores_other_models(connection: Connection) -> None:
    seed(connection, vectors={0: [1.0, 0.0]})
    assert vector_search(connection, [1.0, 0.0], "different-model", limit=5) == []


def test_random_chunks_is_reproducible_for_a_seed(connection: Connection) -> None:
    seed(connection)

    first = random_chunks(connection, "query-a", limit=3)
    again = random_chunks(connection, "query-a", limit=3)

    assert [c.chunk_id for c in first] == [c.chunk_id for c in again]


def test_random_chunks_differs_between_seeds(connection: Connection) -> None:
    seed(connection)

    a = [c.chunk_id for c in random_chunks(connection, "query-a", limit=3)]
    b = [c.chunk_id for c in random_chunks(connection, "query-b", limit=3)]

    assert a != b


def test_pool_unions_all_three_sources(connection: Connection) -> None:
    seed(connection, vectors={i: [1.0 if i == 1 else 0.0, 1.0] for i in range(5)})

    pooled = pool_candidates(
        connection,
        "reader span",
        embed=lambda texts: [[0.0, 1.0] for _ in texts],
        model=MODEL,
        vector_k=2,
        lexical_k=2,
        random_k=2,
    )

    sources = {s for candidate in pooled for s in candidate.sources}
    assert sources == {VECTOR, LEXICAL, RANDOM}


def test_pool_records_every_source_that_found_a_chunk(
    connection: Connection,
) -> None:
    seed(connection, vectors={0: [1.0, 0.0]})

    pooled = pool_candidates(
        connection,
        "reader span passage",
        embed=lambda texts: [[1.0, 0.0] for _ in texts],
        model=MODEL,
        vector_k=1,
        lexical_k=1,
        random_k=0,
    )

    assert len(pooled) == 1
    assert pooled[0].sources == frozenset({VECTOR, LEXICAL})


def test_pool_deduplicates(connection: Connection) -> None:
    seed(connection, vectors={i: [1.0, 0.0] for i in range(5)})

    pooled = pool_candidates(
        connection,
        "the",
        embed=lambda texts: [[1.0, 0.0] for _ in texts],
        model=MODEL,
        vector_k=5,
        lexical_k=5,
        random_k=5,
    )

    ids = [c.chunk.chunk_id for c in pooled]
    assert len(ids) == len(set(ids))


def test_pool_order_is_stable_but_not_rank_order(connection: Connection) -> None:
    seed(connection, vectors={i: [float(5 - i), 1.0] for i in range(5)})

    def pool() -> list[int]:
        return [
            c.chunk.chunk_id
            for c in pool_candidates(
                connection,
                "reader span passage",
                embed=lambda texts: [[5.0, 1.0] for _ in texts],
                model=MODEL,
                vector_k=5,
                lexical_k=0,
                random_k=0,
            )
        ]

    first = pool()
    assert first == pool()

    ranked = [c.chunk_id for c in vector_search(connection, [5.0, 1.0], MODEL, limit=5)]
    assert first != ranked


def test_pool_skips_retrievers_with_zero_k(connection: Connection) -> None:
    seed(connection)

    def explode(texts: list[str]) -> list[list[float]]:
        raise AssertionError("embedder should not be called when vector_k is 0")

    pooled = pool_candidates(
        connection, "reader", explode, MODEL, vector_k=0, lexical_k=3, random_k=0
    )

    assert all(c.sources == frozenset({LEXICAL}) for c in pooled)


def test_lexical_search_matches_on_some_terms_not_all(
    connection: Connection,
) -> None:
    """A natural question matches nothing under AND semantics."""
    seed(connection)

    results = lexical_search(
        connection, "Why does the reader predict a span over kubernetes?", limit=5
    )

    assert results
    assert results[0].ordinal == 0


def test_lexical_search_ranks_more_matching_terms_higher(
    connection: Connection,
) -> None:
    seed(connection)

    results = lexical_search(connection, "reader span embeddings", limit=5)

    # chunk 0 has two of the three terms, chunk 1 has one
    assert results[0].ordinal == 0
    assert 1 in [r.ordinal for r in results]


def test_lexical_search_of_only_stop_words_matches_nothing(
    connection: Connection,
) -> None:
    seed(connection)
    assert lexical_search(connection, "the a of and", limit=5) == []

"""Retrieval over stored chunks, and candidate pooling for labelling.

Three retrievers are provided because the reference set is built by pooling.
Judging only what one retriever returns would make its own recall look perfect
by construction: a chunk it never surfaces could never be labelled relevant.
Drawing candidates from retrievers that see the corpus differently — and from a
random sample that sees it not at all — keeps the resulting measurement honest.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import psycopg

from daedalus.storage.embeddings import _vector_literal

Connection = psycopg.Connection[tuple[object, ...]]
Embedder = Callable[[list[str]], list[list[float]]]

#: Names recorded for the retriever that surfaced a candidate.
VECTOR = "vector"
LEXICAL = "lexical"
RANDOM = "random"

_COLUMNS = "c.id, c.doc_id, c.ordinal, c.kind, c.text, c.heading_path"

_VECTOR_SEARCH = f"""
SELECT {_COLUMNS}
FROM chunks c
JOIN embeddings e ON e.chunk_id = c.id AND e.model = %s
ORDER BY e.embedding <=> %s::vector
LIMIT %s
"""

# The query's lexemes are joined with OR rather than AND. websearch_to_tsquery
# and plainto_tsquery both require every term to be present, so a natural
# question of six content words matches nothing at all — measured at 0 chunks
# against this corpus, where the OR form matched 531. ts_rank still orders
# chunks containing more of the query's terms first, so precision is recovered
# by the ranking rather than by the match.
_LEXICAL_SEARCH = f"""
WITH q AS (
    SELECT coalesce(
        array_to_string(tsvector_to_array(to_tsvector('english', %s)), ' | '),
        ''
    )::tsquery AS query
)
SELECT {_COLUMNS}
FROM chunks c, q
WHERE c.search_vector @@ q.query
ORDER BY ts_rank(c.search_vector, q.query) DESC, c.id
LIMIT %s
"""

_RANDOM_SAMPLE = f"""
SELECT {_COLUMNS}
FROM chunks c
ORDER BY md5(c.id::text || %s)
LIMIT %s
"""


@dataclass(frozen=True)
class Chunk:
    """A chunk as returned by a retriever."""

    chunk_id: int
    doc_id: str
    ordinal: int
    kind: str
    text: str
    heading_path: tuple[str, ...]


@dataclass(frozen=True)
class Candidate:
    """A pooled chunk, recording every retriever that surfaced it."""

    chunk: Chunk
    sources: frozenset[str]


def _rows(cursor: psycopg.Cursor[tuple[object, ...]]) -> list[Chunk]:
    """Build chunks from a cursor positioned on the standard column list."""
    return [
        Chunk(
            chunk_id=cast("int", row[0]),
            doc_id=cast("str", row[1]),
            ordinal=cast("int", row[2]),
            kind=cast("str", row[3]),
            text=cast("str", row[4]),
            heading_path=tuple(cast("list[str]", row[5])),
        )
        for row in cursor.fetchall()
    ]


def vector_search(
    connection: Connection, vector: Sequence[float], model: str, limit: int
) -> list[Chunk]:
    """Return the chunks whose embeddings are nearest the given vector.

    Cosine distance is used because the embeddings are normalised and only
    direction carries meaning. Only embeddings from the named model are
    considered: vectors from different models occupy unrelated spaces and
    comparing across them silently produces nonsense.
    """
    with connection.cursor() as cursor:
        cursor.execute(_VECTOR_SEARCH, (model, _vector_literal(vector), limit))
        return _rows(cursor)


def lexical_search(connection: Connection, query: str, limit: int) -> list[Chunk]:
    """Return the chunks matching the query as full text, best rank first.

    The query is reduced to its lexemes and matched with OR, so a chunk need
    only contain some of the terms. Ranking, not matching, supplies precision.
    A query of nothing but stop words reduces to an empty tsquery and matches
    nothing, which is correct rather than an error.
    """
    with connection.cursor() as cursor:
        cursor.execute(_LEXICAL_SEARCH, (query, limit))
        return _rows(cursor)


def random_chunks(connection: Connection, seed: str, limit: int) -> list[Chunk]:
    """Return an arbitrary but reproducible sample of chunks.

    Ordering by a hash of the chunk id and a seed gives a sample that is stable
    for a given seed without touching the session's random state, so a labelling
    session can be resumed and produce the same candidates.
    """
    with connection.cursor() as cursor:
        cursor.execute(_RANDOM_SAMPLE, (seed, limit))
        return _rows(cursor)


def pool_candidates(
    connection: Connection,
    query: str,
    embed: Embedder,
    model: str,
    vector_k: int = 10,
    lexical_k: int = 10,
    random_k: int = 5,
) -> list[Candidate]:
    """Pool candidates for one query from all three retrievers.

    Returned in an order derived from the query text and chunk id, not from any
    retriever's ranking. Presenting candidates in rank order biases a human
    labeller toward the top of the list, which would contaminate the very
    measurement the reference set exists to provide.
    """
    found: dict[int, tuple[Chunk, set[str]]] = {}

    def add(chunks: Sequence[Chunk], source: str) -> None:
        for chunk in chunks:
            found.setdefault(chunk.chunk_id, (chunk, set()))[1].add(source)

    if vector_k:
        vector = embed([query])[0]
        add(vector_search(connection, vector, model, vector_k), VECTOR)
    if lexical_k:
        add(lexical_search(connection, query, lexical_k), LEXICAL)
    if random_k:
        add(random_chunks(connection, query, random_k), RANDOM)

    candidates = [
        Candidate(chunk=chunk, sources=frozenset(sources))
        for chunk, sources in found.values()
    ]
    return sorted(candidates, key=lambda c: _presentation_key(query, c.chunk.chunk_id))


def _presentation_key(query: str, chunk_id: int) -> str:
    """A stable, rank-independent ordering key for one query's candidates."""
    return hashlib.md5(f"{query}:{chunk_id}".encode()).hexdigest()

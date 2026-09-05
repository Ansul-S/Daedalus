"""Populate the candidates table with the pool every judgement was drawn from.

Run once, after migration 003. Idempotent: every insert is ON CONFLICT DO
NOTHING, so re-running it reports zero new rows and changes nothing.

Two passes.

The original pool is recomputed from the three retrievers that built it. That
recomputation is exact rather than approximate -- the random draw is seeded on
the query text, so the pool regenerates identically as long as the chunks and
their embeddings are unchanged, and this script verifies that it reproduces the
existing judgements before writing anything.

The challenger pools are recomputed from the two retrievers that did not build
it: bge-m3 over chunk text alone, and all-minilm over the same heading-path plus
text input production uses. Neither writes an embedding; the vectors are held in
memory and discarded.

The 1,064 existing judgements are never read for anything but verification and
never written. The script prints a checksum over them before and after, and
refuses to continue if it moves.
"""

from __future__ import annotations

import math
import sys
from typing import cast

import psycopg

from daedalus.embedding import embed_texts
from daedalus.retrieval.search import (
    LEXICAL,
    RANDOM,
    VECTOR,
    lexical_search,
    random_chunks,
    vector_search,
)
from daedalus.storage.database import connect
from daedalus.storage.embeddings import embedding_input
from daedalus.storage.queries import record_candidates

Connection = psycopg.Connection[tuple[object, ...]]

#: The frozen pooling parameters the reference set was built with.
VECTOR_K = 10
LEXICAL_K = 10
RANDOM_K = 5

#: Production embedding model, and the challengers measured against it.
PRODUCTION = "bge-m3"
NOHEADING = "bge-m3-noheading"
MINILM = "all-minilm"

#: Depth taken from each challenger, matching the original pooling depth.
CHALLENGER_K = 10

_CHECKSUM = """
SELECT count(*), md5(string_agg(
    query_id || '|' || doc_id || '|' || ordinal || '|' || grade || '|' || judged_at,
    E'\n' ORDER BY query_id, doc_id, ordinal))
FROM judgements
"""

Row = tuple[int, str, int, str, int | None]


def judgement_checksum(connection: Connection) -> tuple[int, str]:
    """Return the row count and a checksum over every judgement."""
    with connection.cursor() as cursor:
        cursor.execute(_CHECKSUM)
        row = cursor.fetchone()
        return (cast("int", row[0]), cast("str", row[1])) if row else (0, "")


def unit(vector: list[float]) -> list[float]:
    """Return the vector scaled to unit length, so a dot product is cosine."""
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def original_pool(connection: Connection, queries: list[tuple[int, str]]) -> list[Row]:
    """Recompute the vector, lexical and random draws that built the pool."""
    rows: list[Row] = []
    for query_id, text in queries:
        vector = embed_texts([text], model=PRODUCTION)[0]
        for rank, chunk in enumerate(
            vector_search(connection, vector, PRODUCTION, VECTOR_K), start=1
        ):
            rows.append((query_id, chunk.doc_id, chunk.ordinal, VECTOR, rank))
        for rank, chunk in enumerate(
            lexical_search(connection, text, LEXICAL_K), start=1
        ):
            rows.append((query_id, chunk.doc_id, chunk.ordinal, LEXICAL, rank))
        # The random sample has no ranking to preserve; its order is an artefact
        # of the seed hash, not a statement that one chunk beat another.
        for chunk in random_chunks(connection, text, RANDOM_K):
            rows.append((query_id, chunk.doc_id, chunk.ordinal, RANDOM, None))
    return rows


def challenger_pool(
    connection: Connection,
    queries: list[tuple[int, str]],
    label: str,
    model: str,
    with_heading: bool,
) -> list[Row]:
    """Rank every chunk for every query with one challenger, in memory."""
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT doc_id, ordinal, heading_path, text FROM chunks ORDER BY id"
        )
        chunks = cursor.fetchall()

    refs = [(cast("str", c[0]), cast("int", c[1])) for c in chunks]
    inputs = [
        embedding_input(c[2], c[3]) if with_heading else str(c[3]) for c in chunks
    ]
    vectors = []
    for start in range(0, len(inputs), 32):
        vectors.extend(embed_texts(inputs[start : start + 32], model=model))
    vectors = [unit(v) for v in vectors]
    query_vectors = [unit(v) for v in embed_texts([t for _, t in queries], model=model)]

    rows: list[Row] = []
    for (query_id, _), query_vector in zip(queries, query_vectors, strict=True):
        # Ties are broken on (doc_id, ordinal) so the draw is reproducible.
        ranked = sorted(
            (
                (sum(a * b for a, b in zip(query_vector, vector, strict=True)), index)
                for index, vector in enumerate(vectors)
            ),
            key=lambda scored: (-scored[0], refs[scored[1]]),
        )
        for rank, (_, index) in enumerate(ranked[:CHALLENGER_K], start=1):
            rows.append((query_id, refs[index][0], refs[index][1], label, rank))
    return rows


def main() -> int:
    with connect() as connection:
        before = judgement_checksum(connection)
        print(f"judgements before: {before[0]} rows, md5 {before[1]}")

        with connection.cursor() as cursor:
            cursor.execute("SELECT id, text FROM queries ORDER BY id")
            queries = [
                (cast("int", r[0]), cast("str", r[1])) for r in cursor.fetchall()
            ]
            cursor.execute("SELECT query_id, doc_id, ordinal FROM judgements")
            judged: dict[int, set[tuple[str, int]]] = {}
            for query_id, doc_id, ordinal in cursor.fetchall():
                judged.setdefault(cast("int", query_id), set()).add(
                    (cast("str", doc_id), cast("int", ordinal))
                )

        print(f"queries: {len(queries)}")

        original = original_pool(connection, queries)
        # The recomputed pool must account for every existing judgement. If it
        # does not, the corpus or the embeddings have moved and the provenance
        # this script would write would be a guess.
        pooled: dict[int, set[tuple[str, int]]] = {}
        for query_id, doc_id, ordinal, _, _ in original:
            pooled.setdefault(query_id, set()).add((doc_id, ordinal))
        missing = {q: judged[q] - pooled.get(q, set()) for q in judged}
        unaccounted = {q: refs for q, refs in missing.items() if refs}
        if unaccounted:
            print(
                f"ABORT: {len(unaccounted)} queries have judgements the "
                f"recomputed pool does not contain: {sorted(unaccounted)[:5]}"
            )
            return 1
        print(
            f"original pool rows: {len(original)} "
            f"(every one of the {sum(len(v) for v in judged.values())} "
            f"judgements is accounted for)"
        )

        challengers = challenger_pool(
            connection, queries, NOHEADING, PRODUCTION, with_heading=False
        ) + challenger_pool(connection, queries, MINILM, MINILM, with_heading=True)
        print(f"challenger pool rows: {len(challengers)}")

        written = record_candidates(connection, original + challengers)
        connection.commit()
        print(f"rows inserted: {written}")

        after = judgement_checksum(connection)
        print(f"judgements after:  {after[0]} rows, md5 {after[1]}")
        if after != before:
            print("ABORT: the judgements changed. This must never happen.")
            return 1
        print("judgements unchanged")

        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT source, count(*) FROM candidates GROUP BY source "
                "ORDER BY source"
            )
            for source, count in cursor.fetchall():
                print(f"  {source:18} {count}")
            cursor.execute(
                "SELECT count(*) FROM (SELECT DISTINCT query_id, doc_id, "
                "ordinal FROM candidates) c"
            )
            distinct = cursor.fetchone()
            cursor.execute("SELECT count(*) FROM unjudged_candidates")
            unjudged = cursor.fetchone()
        print(f"  distinct candidates {distinct[0] if distinct else 0}")
        print(f"  awaiting judgement  {unjudged[0] if unjudged else 0}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

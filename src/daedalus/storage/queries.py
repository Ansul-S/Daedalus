"""The labelled reference set: queries, candidate pools, and judgements."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import psycopg

Connection = psycopg.Connection[tuple[object, ...]]

#: Where a query came from. Harvested questions are taken from the corpus
#: itself; authored ones are written by the person doing the labelling.
QUERY_SOURCES = ("harvested", "authored")

#: Relevance grades, lowest first.
GRADES = (0, 1, 2)

_INSERT_QUERY = """
INSERT INTO queries (text, source) VALUES (%s, %s)
ON CONFLICT (text) DO NOTHING
RETURNING id
"""

_UPSERT_JUDGEMENT = """
INSERT INTO judgements (query_id, doc_id, ordinal, grade)
VALUES (%s, %s, %s, %s)
ON CONFLICT (query_id, doc_id, ordinal) DO UPDATE
SET grade = EXCLUDED.grade, judged_at = now()
"""

_INSERT_CANDIDATE = """
INSERT INTO candidates (query_id, doc_id, ordinal, source, rank)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (query_id, doc_id, ordinal, source) DO NOTHING
"""


@dataclass(frozen=True)
class Query:
    """A query in the reference set, with how much of it has been judged."""

    query_id: int
    text: str
    source: str
    judged: int


def add_query(connection: Connection, text: str, source: str) -> int | None:
    """Add a query, returning its id, or None if the text is already present.

    Queries are unique by text so that re-running a harvest does not create
    duplicates that would be labelled twice.
    """
    if source not in QUERY_SOURCES:
        raise ValueError(f"source must be one of {QUERY_SOURCES}, got {source!r}")

    with connection.cursor() as cursor:
        cursor.execute(_INSERT_QUERY, (text.strip(), source))
        row = cursor.fetchone()
        return cast("int", row[0]) if row else None


def list_queries(connection: Connection) -> list[Query]:
    """Return every query with the number of judgements recorded against it."""
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT q.id, q.text, q.source, count(j.query_id)
            FROM queries q
            LEFT JOIN judgements j ON j.query_id = q.id
            GROUP BY q.id, q.text, q.source
            ORDER BY q.id
            """
        )
        return [
            Query(
                query_id=cast("int", row[0]),
                text=cast("str", row[1]),
                source=cast("str", row[2]),
                judged=cast("int", row[3]),
            )
            for row in cursor.fetchall()
        ]


def record_judgement(
    connection: Connection, query_id: int, doc_id: str, ordinal: int, grade: int
) -> None:
    """Record a relevance grade, replacing any earlier grade for the same pair."""
    if grade not in GRADES:
        raise ValueError(f"grade must be one of {GRADES}, got {grade!r}")

    with connection.cursor() as cursor:
        cursor.execute(_UPSERT_JUDGEMENT, (query_id, doc_id, ordinal, grade))


def judged_pairs(connection: Connection, query_id: int) -> set[tuple[str, int]]:
    """Return the (doc_id, ordinal) pairs already judged for this query.

    Used to skip candidates on a resumed labelling session.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT doc_id, ordinal FROM judgements WHERE query_id = %s", (query_id,)
        )
        return {(cast("str", row[0]), cast("int", row[1])) for row in cursor.fetchall()}


def grade_totals(connection: Connection) -> dict[int, int]:
    """Return how many judgements exist at each grade."""
    with connection.cursor() as cursor:
        cursor.execute("SELECT grade, count(*) FROM judgements GROUP BY grade")
        return {cast("int", row[0]): cast("int", row[1]) for row in cursor.fetchall()}


def record_candidates(
    connection: Connection, rows: Sequence[tuple[int, str, int, str, int | None]]
) -> int:
    """Record candidate provenance as (query_id, doc_id, ordinal, source, rank).

    Rows already present are left untouched rather than updated, so recording
    the same pool twice is a no-op and a backfill can be re-run safely. The rank
    is the position the source surfaced the candidate at, or None for a source
    that has no meaningful order, such as the random sample.

    Returns the number of rows actually inserted.
    """
    if not rows:
        return 0
    with connection.cursor() as cursor:
        cursor.executemany(_INSERT_CANDIDATE, rows)
        return cursor.rowcount


def candidate_refs(connection: Connection, query_id: int) -> set[tuple[str, int]]:
    """Return the distinct (doc_id, ordinal) pairs recorded for this query.

    Distinct, because one chunk may have been surfaced by several retrievers and
    is still only one thing to judge.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT DISTINCT doc_id, ordinal FROM candidates WHERE query_id = %s",
            (query_id,),
        )
        return {(cast("str", row[0]), cast("int", row[1])) for row in cursor.fetchall()}


def unjudged_candidate_counts(connection: Connection) -> dict[int, int]:
    """Return how many recorded candidates still lack a judgement, per query.

    An empty result means the recorded pool is fully judged. This replaces
    recomputing every pool and comparing, which needed the embedding model and
    could not see candidates the original retrievers no longer produce.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT query_id, count(*) FROM unjudged_candidates GROUP BY query_id"
        )
        return {cast("int", row[0]): cast("int", row[1]) for row in cursor.fetchall()}

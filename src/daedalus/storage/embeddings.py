"""Storing embeddings for chunks, and backfilling the ones that lack them.

Embedding runs separately from ingestion so that material can be re-embedded
with a different model without being re-parsed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import psycopg

#: Chunks embedded per request to the embedding service.
DEFAULT_BATCH_SIZE = 32

_MISSING = """
SELECT c.id, c.heading_path, c.text
FROM chunks c
LEFT JOIN embeddings e ON e.chunk_id = c.id AND e.model = %s
WHERE e.chunk_id IS NULL
ORDER BY c.id
LIMIT %s
"""

_UPSERT = """
INSERT INTO embeddings (chunk_id, model, dim, embedding)
VALUES (%s, %s, %s, %s::vector)
ON CONFLICT (chunk_id, model) DO UPDATE
SET embedding = EXCLUDED.embedding, dim = EXCLUDED.dim, created_at = now()
"""

Embedder = Callable[[list[str]], list[list[float]]]
Connection = psycopg.Connection[tuple[object, ...]]


def embedding_input(heading_path: Sequence[str], text: str) -> str:
    """Return the text to embed for a chunk.

    The heading path is prepended so that a short chunk — an output of a few
    tokens, say — carries the context of the section it belongs to. Only the
    embedded text is affected; the stored text is untouched, so embedding with
    and without this prefix can be compared without re-chunking.
    """
    if not heading_path:
        return text
    return f"{' > '.join(heading_path)}\n\n{text}"


def chunks_missing_embeddings(
    connection: Connection, model: str, limit: int
) -> list[tuple[int, list[str], str]]:
    """Return chunks that have no embedding for this model, lowest id first."""
    with connection.cursor() as cursor:
        cursor.execute(_MISSING, (model, limit))
        # psycopg returns id as int, heading_path as list[str], text as str;
        # the cursor's element type is not narrow enough to say so.
        return [cast("tuple[int, list[str], str]", row) for row in cursor.fetchall()]


def store_embeddings(
    connection: Connection, model: str, vectors: Sequence[tuple[int, list[float]]]
) -> int:
    """Write embeddings, replacing any existing vector for the same model.

    Returns the number of rows written.
    """
    rows = [
        (chunk_id, model, len(vector), _vector_literal(vector))
        for chunk_id, vector in vectors
    ]
    with connection.cursor() as cursor:
        cursor.executemany(_UPSERT, rows)
    return len(rows)


def backfill_embeddings(
    connection: Connection,
    embed: Embedder,
    model: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Embed every chunk lacking an embedding for this model.

    The embedder is injected rather than imported so that the storage layer
    holds no knowledge of the embedding service.

    Each batch is committed as it completes, so an interrupted run keeps the
    work it finished and a later run resumes from there.

    Returns the total number of chunks embedded.
    """
    total = 0
    while True:
        batch = chunks_missing_embeddings(connection, model, batch_size)
        if not batch:
            return total

        inputs = [embedding_input(heading, text) for _, heading, text in batch]
        vectors = embed(inputs)
        total += store_embeddings(
            connection,
            model,
            list(zip([row[0] for row in batch], vectors, strict=True)),
        )
        connection.commit()


def _vector_literal(vector: Sequence[float]) -> str:
    """Render a vector in the text form pgvector accepts."""
    return f"[{','.join(repr(float(value)) for value in vector)}]"

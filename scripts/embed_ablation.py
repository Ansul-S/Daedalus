"""Embed the corpus for the two ablation retrievers, under their own identities.

`daedalus embed` deliberately does not do this. It uses one name as both the
model to call and the key to store under, and it always prepends the heading
path. Both are right for production and wrong for an ablation:

  all-minilm        a different model, the same input construction
  bge-m3-noheading  the same model, a different input construction

The second is the one the production command cannot express. It calls bge-m3 but
must not be stored as `bge-m3`, because that key holds the vectors the reference
pool was drawn from. Overwriting them would silently invalidate the pool's
provenance, so this script refuses to write under that key at all.

Resumable: only chunks lacking an embedding for the label are embedded, and each
batch is committed as it completes.
"""

from __future__ import annotations

import sys
from typing import cast

import psycopg

from daedalus.embedding import embed_texts
from daedalus.storage.database import connect
from daedalus.storage.embeddings import (
    chunks_missing_embeddings,
    embedding_input,
    store_embeddings,
)

Connection = psycopg.Connection[tuple[object, ...]]

#: The production key. Never written by this script.
PRODUCTION = "bge-m3"

BATCH = 32

#: (storage label, model to call, whether to prepend the heading path)
ABLATIONS = (
    ("bge-m3-noheading", "bge-m3", False),
    ("all-minilm", "all-minilm", True),
)


def production_checksum(connection: Connection) -> str:
    """A checksum over the production vectors, to prove they did not move."""
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT md5(string_agg(chunk_id || ':' || md5(embedding::text), "
            "E'\n' ORDER BY chunk_id)) FROM embeddings WHERE model = %s",
            (PRODUCTION,),
        )
        row = cursor.fetchone()
        return cast("str", row[0]) if row and row[0] else "(none)"


def embed_ablation(
    connection: Connection, label: str, model: str, with_heading: bool
) -> int:
    """Embed every chunk lacking a vector for this label. Returns rows written."""
    if label == PRODUCTION:
        raise ValueError(
            f"refusing to write under {PRODUCTION!r}: that key holds the vectors "
            f"the reference pool was drawn from"
        )

    total = 0
    while True:
        batch = chunks_missing_embeddings(connection, label, BATCH)
        if not batch:
            return total
        inputs = [
            embedding_input(heading, text) if with_heading else text
            for _, heading, text in batch
        ]
        vectors = embed_texts(inputs, model=model)
        total += store_embeddings(
            connection,
            label,
            list(zip([row[0] for row in batch], vectors, strict=True)),
        )
        connection.commit()
        print(f"    {total} embedded", end="\r", flush=True)


def main() -> int:
    with connect() as connection:
        before = production_checksum(connection)
        print(f"production {PRODUCTION} checksum before: {before}")

        for label, model, with_heading in ABLATIONS:
            source = "heading path + text" if with_heading else "text only"
            print(f"\n{label}: calling {model} over {source}")
            written = embed_ablation(connection, label, model, with_heading)
            print(f"    wrote {written} embeddings")

        after = production_checksum(connection)
        print(f"\nproduction {PRODUCTION} checksum after : {after}")
        if after != before:
            print("ABORT: the production embeddings changed. This must never happen.")
            return 1
        print("production embeddings unchanged")

        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT model, count(*), min(dim) FROM embeddings "
                "GROUP BY model ORDER BY model"
            )
            for model, count, dim in cursor.fetchall():
                print(f"  {str(model):18} {count} vectors, dim {dim}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Command line entry point.

Ingestion and embedding are separate commands because they have very different
costs: parsing a notebook takes milliseconds, embedding its chunks takes tens of
seconds. Keeping them apart means re-ingesting never forces re-embedding.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import psycopg

from daedalus.embedding import DEFAULT_MODEL, EmbeddingError, embed_texts
from daedalus.ingestion.canonical import notebook_to_document
from daedalus.ingestion.notebook import parse_notebook
from daedalus.labelling import LABEL_PASSES, cmd_label, cmd_label_questions
from daedalus.storage.database import DatabaseNotConfiguredError, connect
from daedalus.storage.documents import (
    document_exists,
    store_document,
)
from daedalus.storage.embeddings import DEFAULT_BATCH_SIZE, backfill_embeddings
from daedalus.storage.queries import (
    QUERY_SOURCES,
    add_query,
    list_queries,
)

Connection = psycopg.Connection[tuple[object, ...]]

#: File extensions the ingester recognises, mapped to nothing yet beyond
#: notebooks. Other formats join this as their parsers are written.
NOTEBOOK_SUFFIX = ".ipynb"

#: Keystrokes accepted while labelling, mapped to the grade they record.
#: A grade enters the reference set only through one of these keystrokes.


def notebook_paths(paths: Sequence[Path]) -> list[Path]:
    """Expand the given paths into notebook files, sorted for reproducibility.

    A directory contributes the notebooks directly inside it. A file is taken
    as given, and is an error if it is not a notebook.
    """
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(path.glob(f"*{NOTEBOOK_SUFFIX}")))
        elif path.suffix == NOTEBOOK_SUFFIX:
            found.append(path)
        else:
            raise ValueError(f"not a notebook: {path}")
    return found


def cmd_ingest(args: argparse.Namespace) -> int:
    """Parse notebooks and store their chunks.

    A document whose identifier is already stored is skipped, because the
    identifier is derived from content and so an unchanged file would produce
    identical chunks. Storing it again would delete those chunks and, through
    the cascade, discard their embeddings — work that costs orders of magnitude
    more than parsing. Pass --force to store regardless.
    """
    paths = notebook_paths(args.paths)
    if not paths:
        print("no notebooks found", file=sys.stderr)
        return 1

    stored = skipped = 0
    with connect() as connection:
        for path in paths:
            document = notebook_to_document(parse_notebook(path))

            if not args.force and document_exists(connection, document.doc_id):
                print(f"{path.name}: unchanged, skipped  [{document.doc_id}]")
                skipped += 1
                continue

            count = store_document(connection, document)
            print(f"{path.name}: {count} chunks  [{document.doc_id}]")
            stored += 1
        connection.commit()

    print(f"stored {stored}, skipped {skipped}")
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    """Embed every stored chunk that lacks an embedding for the model."""
    with connect() as connection:
        total = backfill_embeddings(
            connection,
            lambda texts: embed_texts(texts, model=args.model),
            args.model,
            batch_size=args.batch_size,
        )
    print(f"embedded {total} chunks with {args.model}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Report what is stored, and how much of it is embedded."""
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM documents")
        documents = cursor.fetchone()
        cursor.execute("SELECT count(*) FROM chunks")
        chunks = cursor.fetchone()
        cursor.execute(
            "SELECT model, count(*), min(dim) FROM embeddings GROUP BY model ORDER BY 1"
        )
        per_model = cursor.fetchall()

    print(f"documents: {documents[0] if documents else 0}")
    print(f"chunks:    {chunks[0] if chunks else 0}")
    if not per_model:
        print("embeddings: none")
    for model, count, dim in per_model:
        print(f"embeddings: {count} with {model} ({dim} dimensions)")
    return 0


def cmd_query_add(args: argparse.Namespace) -> int:
    """Add a query to the reference set."""
    with connect() as connection:
        query_id = add_query(connection, args.text, args.source)
        connection.commit()

    if query_id is None:
        print("query already present, not added")
        return 0
    print(f"added query {query_id}")
    return 0


def cmd_query_list(args: argparse.Namespace) -> int:
    """List queries and how many judgements each has."""
    with connect() as connection:
        queries = list_queries(connection)

    if not queries:
        print("no queries yet")
        return 0
    for query in queries:
        print(
            f"{query.query_id:>4}  {query.source:<9} {query.judged:>3} judged  "
            f"{query.text}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for every subcommand."""
    parser = argparse.ArgumentParser(prog="daedalus", description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    ingest = subcommands.add_parser("ingest", help="parse and store notebooks")
    ingest.add_argument("paths", nargs="+", type=Path, help="files or directories")
    ingest.add_argument(
        "--force",
        action="store_true",
        help="store even if the document is already present, discarding its embeddings",
    )
    ingest.set_defaults(handler=cmd_ingest)

    embed = subcommands.add_parser("embed", help="embed chunks lacking an embedding")
    embed.add_argument("--model", default=DEFAULT_MODEL)
    embed.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    embed.set_defaults(handler=cmd_embed)

    status = subcommands.add_parser("status", help="report what is stored")
    status.set_defaults(handler=cmd_status)

    query = subcommands.add_parser("query", help="manage reference-set queries")
    query_actions = query.add_subparsers(dest="query_command", required=True)

    query_add = query_actions.add_parser("add", help="add a query")
    query_add.add_argument("text")
    query_add.add_argument("--source", choices=QUERY_SOURCES, required=True)
    query_add.set_defaults(handler=cmd_query_add)

    query_list = query_actions.add_parser("list", help="list queries")
    query_list.set_defaults(handler=cmd_query_list)

    label = subcommands.add_parser("label", help="judge pooled candidates")
    label.add_argument("--model", default=DEFAULT_MODEL)
    label.add_argument("--vector-k", type=int, default=10)
    label.add_argument("--lexical-k", type=int, default=10)
    label.add_argument("--random-k", type=int, default=5)
    label.add_argument(
        "--per-query",
        type=int,
        default=25,
        help="record at most this many judgements for one query per session",
    )
    label.add_argument(
        "--regrade",
        type=int,
        metavar="QUERY_ID",
        help="offer one query's whole pool again, replacing its grades",
    )
    label.set_defaults(handler=cmd_label)

    questions = subcommands.add_parser(
        "label-questions", help="label generated questions for one rubric"
    )
    questions.add_argument("rubric", choices=tuple(LABEL_PASSES))
    questions.add_argument(
        "--limit",
        type=int,
        default=10**9,
        help="stop after this many labels in one session",
    )
    questions.set_defaults(handler=cmd_label_questions)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a subcommand, turning expected failures into a message and status 1."""
    args = build_parser().parse_args(argv)
    try:
        exit_code: int = args.handler(args)
        return exit_code
    except (DatabaseNotConfiguredError, EmbeddingError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

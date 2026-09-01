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

from daedalus.embedding import DEFAULT_MODEL, EmbeddingError, embed_texts
from daedalus.ingestion.canonical import notebook_to_document
from daedalus.ingestion.notebook import parse_notebook
from daedalus.retrieval.search import Candidate, pool_candidates
from daedalus.storage.database import DatabaseNotConfiguredError, connect
from daedalus.storage.documents import document_exists, parent_text, store_document
from daedalus.storage.embeddings import DEFAULT_BATCH_SIZE, backfill_embeddings
from daedalus.storage.queries import (
    QUERY_SOURCES,
    add_query,
    grade_totals,
    judged_pairs,
    list_queries,
    record_judgement,
)

#: File extensions the ingester recognises, mapped to nothing yet beyond
#: notebooks. Other formats join this as their parsers are written.
NOTEBOOK_SUFFIX = ".ipynb"

#: Keystrokes accepted while labelling, mapped to the grade they record.
#: A grade enters the reference set only through one of these keystrokes.
GRADE_KEYS = {"0": 0, "1": 1, "2": 2}

#: Characters of a chunk shown before it is truncated. Chosen against the
#: corpus: median chunk length is about 418 characters and p90 about 1,188, so
#: this shows most chunks whole while capping the rare very large one. The full
#: text is always available with the "f" key, because the policy is to judge the
#: complete content and never the preview alone.
PREVIEW_LIMIT = 2000

#: One-line meaning of each grade, shown on every prompt.
GRADE_MEANINGS = (
    "0 not relevant",
    "1 partially answers",
    "2 fully answers",
)

#: Shown once at the start of a session and again on demand with "?".
POLICY_REMINDER = """\
Judge only how useful this chunk's content is for answering the query.

  0  not relevant       related topic or shared words, but does not help answer
  1  partially answers  contributes part of the answer, or evidence for it
  2  fully answers      content is sufficient to answer the query directly

Code and prose are judged by the same standard. Do not downgrade a chunk for
being code, and do not promote it for looking sophisticated. A signature with no
meaningful body does not earn a 2 because its name matches the query.

Ignore how the chunk was retrieved. The question is only whether the chunk
helps answer the query.

A truncated chunk is marked TRUNCATED; press f to read all of it before
judging. When an output chunk is shown, the code that produced it appears above
as context only — the grade belongs to the output, not to that code.

s skips without recording anything; the candidate returns in a later session.

Full policy: docs/LABELLING.md
"""


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


def read_key(prompt: str) -> str:
    """Read a single keypress, falling back to a line when stdin is not a tty.

    A single keypress matters here: a labelling session runs to a thousand or
    more judgements, and requiring Enter on each doubles the effort. The line
    fallback keeps the loop testable and usable through a pipe.
    """
    print(prompt, end="", flush=True)
    if not sys.stdin.isatty():
        return sys.stdin.readline().strip()[:1]

    import termios
    import tty

    descriptor = sys.stdin.fileno()
    saved = termios.tcgetattr(descriptor)
    try:
        tty.setraw(descriptor)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, saved)
    print(key)
    return key


def show_candidate(
    query_text: str,
    candidate: Candidate,
    position: int,
    total: int,
    full: bool = False,
    context: str | None = None,
) -> None:
    """Print one candidate for judging.

    Which retrievers surfaced the candidate is deliberately not shown. The
    reference set measures those retrievers, so a judgement influenced by them
    would be measuring itself.

    An output chunk is often meaningless read alone, so the code that produced
    it is shown above as context. It is labelled as context and excluded from
    the judgement: the grade belongs to the candidate.
    """
    chunk = candidate.chunk
    heading = " > ".join(chunk.heading_path) or "(no heading)"

    print("\n" + "=" * 78)
    print(f"QUERY: {query_text}")
    print(f"[{position}/{total}]  {chunk.kind}  {chunk.doc_id}:{chunk.ordinal}")
    print(f"SECTION: {heading}")

    if context is not None:
        print("-" * 78)
        print("CONTEXT — the code that produced this output. NOT judged.")
        print("-" * 78)
        print(_body(context, full))

    print("-" * 78)
    if context is not None:
        print("CANDIDATE — judge this:")
        print("-" * 78)
    print(_body(chunk.text, full))
    print("-" * 78)


def _body(text: str, full: bool) -> str:
    """Render chunk text, marking truncation explicitly when it applies."""
    body = text.strip()
    if full or len(body) <= PREVIEW_LIMIT:
        return body
    hidden = len(body) - PREVIEW_LIMIT
    return (
        f"{body[:PREVIEW_LIMIT]}\n"
        f"[TRUNCATED — {hidden} more characters. Press f to read all before judging.]"
    )


def cmd_label(args: argparse.Namespace) -> int:
    """Judge pooled candidates for each query, one at a time.

    Candidates already judged are skipped, so a session can be stopped and
    resumed. Each judgement is committed as it is made, for the same reason.

    ``--regrade`` names one query and offers its whole pool again, including
    candidates already graded, so a judgement made in error can be replaced.
    The earlier grade is not shown: the second reading has to stand on its own.
    """
    with connect() as connection:
        if args.regrade is not None:
            queries = [
                q for q in list_queries(connection) if q.query_id == args.regrade
            ]
            if not queries:
                print(f"no query with id {args.regrade}")
                return 1
        else:
            queries = [q for q in list_queries(connection) if q.judged < args.per_query]
        if not queries:
            print("nothing to label")
            return 0

        print(POLICY_REMINDER)

        for query in queries:
            done: set[tuple[str, int]] = set()
            if args.regrade is None:
                done = judged_pairs(connection, query.query_id)
            pooled = pool_candidates(
                connection,
                query.text,
                lambda texts: embed_texts(texts, model=args.model),
                args.model,
                vector_k=args.vector_k,
                lexical_k=args.lexical_k,
                random_k=args.random_k,
            )
            pending = [
                c for c in pooled if (c.chunk.doc_id, c.chunk.ordinal) not in done
            ]

            for position, candidate in enumerate(pending, start=1):
                context = parent_text(
                    connection, candidate.chunk.doc_id, candidate.chunk.ordinal
                )
                full = False
                while True:
                    show_candidate(
                        query.text,
                        candidate,
                        position,
                        len(pending),
                        full,
                        context,
                    )
                    prompt = (
                        "  ".join(GRADE_MEANINGS)
                        + "   s skip   f full text   ? policy   q quit > "
                    )
                    key = read_key(prompt)

                    if key == "q":
                        print("\nstopped")
                        return 0
                    if key == "?":
                        print("\n" + POLICY_REMINDER)
                        continue
                    if key == "f":
                        full = True
                        continue
                    break

                if key not in GRADE_KEYS:
                    continue

                record_judgement(
                    connection,
                    query.query_id,
                    candidate.chunk.doc_id,
                    candidate.chunk.ordinal,
                    GRADE_KEYS[key],
                )
                connection.commit()

        totals = grade_totals(connection)

    print(
        "\njudgements by grade: "
        + ", ".join(
            f"{grade}={totals.get(grade, 0)}" for grade in sorted(GRADE_KEYS.values())
        )
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
        help="stop offering a query once it has this many judgements",
    )
    label.add_argument(
        "--regrade",
        type=int,
        metavar="QUERY_ID",
        help="offer one query's whole pool again, replacing its grades",
    )
    label.set_defaults(handler=cmd_label)

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

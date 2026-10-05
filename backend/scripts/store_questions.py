"""Put the questions a review keeps into the library.

Run from backend/:  uv run python -m scripts.store_questions [--dry-run] [--replace]

The review's choice is data/inventory/<database>.library.json: each question it keeps names the
questions file beside the inventory it was written into and the idea it asks about, with the
review's verdict, and a fixable one the edit that fixes it. Only a question that passed every
check goes in, accepted, with the passages it was written from as its sources; its edit is made
as the question bank makes one and kept in its report. A question already in is left as it is,
so the same command can run again. The local embedding model embeds each question for the
duplicate check, so Ollama has to be running.

--replace takes out the questions no review put in -- a library written before -- in the same
transaction, and refuses while any of them has been answered, scheduled, reviewed or rated.
--dry-run shows what would happen and changes nothing.
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.engine import make_url

from app.core.checks import check_ollama
from app.core.config import Settings, get_settings
from app.db.models import Question, Topic
from app.db.session import SessionFactory, engine
from app.llm.models import embedding_model
from app.llm.tracing import tracing
from app.questions.generation import Source
from app.questions.inventory import (
    DocumentInventory,
    Review,
    from_json,
    load_document,
    resume,
)
from app.questions.library import (
    Kept,
    Selection,
    earlier,
    in_library,
    passage_topic,
    practised,
    store,
    unfit,
)
from app.questions.topics import topic_for
from app.questions.writing import Entry, read_entries, sources_of
from scripts.ingest import configure_logging
from scripts.inventory import review_path, reviewed


def library_path(inventory: Path) -> Path:
    """Where a review's choice for the library is kept: data/inventory/<database>.library.json"""
    return inventory.with_name(f"{inventory.stem}.library.json")


async def needs_ollama(settings: Settings) -> list[str]:
    """What stops the embedding model from embedding the questions, if anything."""
    wanted = {"ollama", f"model {settings.embedding_model}"}
    return [
        f"{check.name}: {check.detail}"
        for check in await check_ollama(settings)
        if check.status == "fail" and check.name in wanted
    ]


@dataclass
class Saved:
    """What storing reads from the files beside the inventory."""

    documents: list[DocumentInventory]
    review: Review | None
    # The questions of each file the review names, by idea
    files: dict[str, dict[str, Entry]]


def read_saved(inventory_path: Path, selection: Selection) -> Saved:
    documents, _, _ = from_json(json.loads(inventory_path.read_text()))
    review = None
    if review_path(inventory_path).exists():
        review = Review.model_validate_json(review_path(inventory_path).read_text())
    files: dict[str, dict[str, Entry]] = {}
    for name in dict.fromkeys(kept.file for kept in selection.questions):
        path = inventory_path.parent / name
        entries = read_entries(json.loads(path.read_text())) if path.exists() else []
        files[name] = {entry.key: entry for entry in entries}
    return Saved(documents, review, files)


async def planned(
    saved: Saved, selection: Selection
) -> tuple[list[tuple[Kept, Entry, list[Source]]], list[str]]:
    """Each kept question with the entry it was written as and its idea's passages, and what
    stops any of them from going in."""
    async with SessionFactory() as session:
        documents = []
        for read in saved.documents:
            document = await load_document(session, read.document_id)
            if document is None or not resume(document, read):
                return [], [f"document {read.document_id} has changed since the inventory read it"]
            documents.append(document)
    concepts = reviewed(documents, saved.review)

    plans: list[tuple[Kept, Entry, list[Source]]] = []
    problems: list[str] = []
    for kept in selection.questions:
        entry = saved.files[kept.file].get(kept.key)
        concept = concepts.get(kept.key)
        if entry is None:
            problems.append(f"{kept.key}: {kept.file} has no question on it")
        elif concept is None:
            problems.append(f"{kept.key}: the inventory no longer has this idea")
        else:
            sources = sources_of(concept, documents)
            if (why := unfit(entry, sources)) is not None:
                problems.append(f"{kept.key} in {kept.file}: {why}")
            else:
                plans.append((kept, entry, sources))
    return plans, problems


async def main(args: argparse.Namespace) -> int:
    settings = get_settings()
    if settings.environment != "local" or settings.fake_models:
        print("The library is filled locally, on real models (ENVIRONMENT=local, no FAKE_MODELS).")
        return 1
    database = make_url(settings.database_url).database
    inventory_path = settings.data_dir / "inventory" / f"{database}.json"
    path = args.library or library_path(inventory_path)
    if not inventory_path.exists() or not path.exists():
        print(f"Storing needs the inventory at {inventory_path} and the review's choice at {path}.")
        return 1
    selection = Selection.model_validate_json(path.read_text())
    saved = read_saved(inventory_path, selection)
    try:
        return await run(args, settings, path, saved, selection)
    finally:
        await engine.dispose()


async def run(
    args: argparse.Namespace, settings: Settings, path: Path, saved: Saved, selection: Selection
) -> int:
    plans, problems = await planned(saved, selection)
    if problems:
        print("Nothing is stored:", *problems, sep="\n  ")
        return 1

    async with SessionFactory() as session:
        there = await in_library(session)
        old = await earlier(session) if args.replace else []
        held = await practised(session, old)
        print(f"Into {make_url(settings.database_url).database}, as {path} keeps them:")
        wrong: list[str] = []
        for kept, entry, _ in plans:
            topic_id = await topic_for(session, entry.chunk_ids)
            topic = await session.scalar(select(Topic.name).where(Topic.id == topic_id))
            if kept.topic is not None:
                try:
                    await passage_topic(session, entry.chunk_ids, kept.topic)
                except ValueError as exc:
                    wrong.append(f"{kept.key}: {exc}")
                topic = f"{kept.topic} (the review's; {topic} by its passages)"
            state = f"  (in already, question {there[kept.key]})" if kept.key in there else ""
            edit = "  + edit" if kept.edit is not None else ""
            verdict = kept.review.verdict
            print(f"  {kept.key:<6} {verdict:<4} {entry.style:<14} {topic}{edit}{state}")
        if args.replace:
            print(f"Taken out: {len(old)} question(s) no review put in.")
    if held:
        print(f"Nothing is stored: questions {', '.join(map(str, held))} have been practised.")
        return 1
    if wrong:
        print("Nothing is stored:", *wrong, sep="\n  ")
        return 1
    to_store = [plan for plan in plans if plan[0].key not in there]
    if args.dry_run:
        print(f"Would store {len(to_store)} question(s).")
        return 0
    if problems := await needs_ollama(settings):
        print("Embedding the questions needs the local model:", *problems, sep="\n  ")
        return 1

    async with embedding_model(settings) as embedder, SessionFactory() as session:
        async with session.begin():
            if old:
                await session.execute(delete(Question).where(Question.id.in_(old)))
            stored = [
                await store(session, entry, sources, kept, embedder)
                for kept, entry, sources in to_store
            ]
        print(f"Stored {len(stored)} question(s): {', '.join(str(q.id) for q in stored) or '-'}.")
    return 0


def parse_args_from(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--replace",
        action="store_true",
        help="take out the questions no review put in, in the same transaction",
    )
    parser.add_argument("--dry-run", action="store_true", help="show what would happen")
    parser.add_argument("--library", type=Path, metavar="PATH", help="the review's choice")
    parser.add_argument("--verbose", action="store_true", help="show library log messages")
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args_from()
    configure_logging(arguments.verbose)
    with tracing(get_settings(), "store"):
        sys.exit(asyncio.run(main(arguments)))

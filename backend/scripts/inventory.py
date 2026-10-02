"""Read documents into a concept inventory: the ideas each one explains, listed once.

Run from backend/:  uv run python -m scripts.inventory --document 4 --document 7 [--dry-run]

The documents come from the database DATABASE_URL names, and nothing is written to it. Every
answer is saved to data/inventory/<database>.json as it arrives, so a run that stops -- on
Ctrl+C, or on Groq's day running out -- carries on where it stopped when started again.
--dry-run calls no model: it shows the passages held back, the windows and the likely cost.

A person's review of the ideas sits beside the inventory, in <database>.review.json, and is
applied to every save. --show calls no model either: it works the ideas out again from the
saved answers, under today's checks and the review, and saves and shows them.
"""

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

from sqlalchemy.engine import make_url

from app.core.checks import check_ollama
from app.core.config import Settings, get_settings
from app.db.session import SessionFactory, engine
from app.llm.models import embedding_model, paced_generation_model
from app.llm.tracing import tracing
from app.questions.batch import out_of_budget, spent_today
from app.questions.inventory import (
    INSTRUCTIONS,
    MERGE_INSTRUCTIONS,
    Call,
    Comparison,
    Concept,
    DocumentInventory,
    Review,
    apply_review,
    carry_verdicts,
    close_pairs,
    compare,
    from_json,
    keyed,
    load_document,
    merge_document,
    read_window,
    resume,
    to_json,
    total_usage,
    window_request,
)
from scripts.ingest import configure_logging

# What a dry run assumes for what the model writes back, which only a real run can tell: a
# window's ideas at about a hundred tokens each with the reasoning before them, and a merge's
# groups. The prompts are counted as the pacer counts them, a token to four characters.
ANSWER_PER_WINDOW = (1_000, 2_500)
MERGE_ANSWER = 600
# Each idea as a merge request shows it: a number, a name and a one-sentence summary
MERGE_ENTRY = 45
IDEAS_PER_WINDOW = 12


def say(message: str) -> None:
    print(f"  {message}", flush=True)


def show_plan(documents: list[DocumentInventory]) -> None:
    for document in documents:
        passages = len(document.held_back) + sum(len(w.passages) for w in document.windows)
        reasons = Counter(document.held_back.values())
        print(f"\n{document.title} (document {document.document_id})")
        held = ", ".join(f"{reason} {count}" for reason, count in reasons.most_common())
        held = f": {held}" if held else ""
        print(f"  held back {len(document.held_back)} of {passages} passages{held}")
        if document.held_back:
            print(f"    chunks {', '.join(str(chunk) for chunk in document.held_back)}")
        sizes = " + ".join(f"{window.tokens:,}" for window in document.windows)
        print(f"  read in {len(document.windows)} window(s) of {sizes} tokens")
        for number, window in enumerate(document.windows, start=1):
            done = f" (read, {window.call.version})" if window.call is not None else ""
            print(f"    {number}: chunks {window.chunk_ids[0]}-{window.chunk_ids[-1]}{done}")


def estimate(documents: list[DocumentInventory]) -> tuple[int, int, int]:
    """Prompt tokens still to send, and the answers' low and high guesses."""
    prompts, low, high = 0, 0, 0
    for document in documents:
        unread = [window for window in document.windows if window.reading is None]
        for window in unread:
            prompts += len(INSTRUCTIONS + window_request(document.title, window.passages)) // 4
            low, high = low + ANSWER_PER_WINDOW[0], high + ANSWER_PER_WINDOW[1]
        if len(document.windows) > 1 and document.merge is None:
            entries = IDEAS_PER_WINDOW * len(document.windows)
            prompts += len(MERGE_INSTRUCTIONS) // 4 + entries * MERGE_ENTRY
            low, high = low + MERGE_ANSWER, high + MERGE_ANSWER
    return prompts, low, high


def reviewed(documents: list[DocumentInventory], review: Review | None) -> dict[str, Concept]:
    """Every idea, as the review leaves it when there is one."""
    concepts = keyed(documents)
    return apply_review(concepts, review) if review is not None else concepts


def evidence_shown(concept: Concept) -> str:
    held = sum(check.grounded for check in concept.evidence)
    if held == len(concept.evidence):
        return ""
    return "  EVIDENCE NOT FOUND" if held == 0 else f"  evidence {held} of {len(concept.evidence)}"


def show_inventory(
    documents: list[DocumentInventory], concepts: dict[str, Concept], pairs: list[Comparison]
) -> None:
    for document in documents:
        found = [(key, c) for key, c in concepts.items() if c.document_id == document.document_id]
        readings = len(document.readings())
        print(f"\n{document.title}: {len(found)} ideas from {readings} readings")
        for key, concept in found:
            print(
                f"  {key:<6} {concept.interview} {concept.scope:<8} {concept.reason:<6} "
                f"{','.join(concept.kinds):<28} {concept.name}"
                f"  [chunks {', '.join(map(str, concept.chunk_ids)) or '-'}]"
                f"{evidence_shown(concept)}"
                f"{f'  EXCLUDED: {concept.excluded}' if concept.excluded else ''}"
            )
    sentences = [check for concept in concepts.values() for check in concept.evidence]
    unplaced = sum(not any(check.grounded for check in c.evidence) for c in concepts.values())
    print(
        f"\nEvidence: {sum(check.grounded for check in sentences)} of {len(sentences)} "
        f"sentences hold up; {unplaced} of {len(concepts)} ideas have none that does."
    )
    if pairs:
        print(f"\nClose across documents: {len(pairs)} pair(s)")
        for pair in pairs:
            verdict = {True: "same", False: "different", None: "not judged"}[pair.same]
            print(f"  {pair.similarity:.3f} {verdict:<10} {pair.first} {pair.names[0]}")
            print(f"  {'':<17} {pair.second} {pair.names[1]}")


def save(
    path: Path,
    documents: list[DocumentInventory],
    pairs: list[Comparison],
    calls: list[Call],
    review: Review | None = None,
) -> None:
    """Written whole to a temporary file and moved into place, so a stop mid-write never
    leaves half an inventory. The answers are saved even when the review no longer fits them."""
    try:
        data = to_json(documents, pairs, calls, review)
    except ValueError as exc:
        print(f"  The review no longer fits ({exc}): saved without it.")
        data = to_json(documents, pairs, calls)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    temporary.replace(path)


def review_path(path: Path) -> Path:
    """Where the review of the inventory at `path` is kept: data/inventory/<database>.review.json"""
    return path.with_name(f"{path.stem}.review.json")


async def needs_ollama(settings: Settings) -> list[str]:
    """What stops the embedding model from comparing ideas across documents, if anything."""
    wanted = {"ollama", f"model {settings.embedding_model}"}
    return [
        f"{check.name}: {check.detail}"
        for check in await check_ollama(settings)
        if check.status == "fail" and check.name in wanted
    ]


async def main(args: argparse.Namespace) -> int:
    settings = get_settings()
    if settings.environment != "local" or settings.fake_models:
        print("The inventory runs locally, on real models (ENVIRONMENT=local, no FAKE_MODELS).")
        return 1
    database = make_url(settings.database_url).database
    path = args.out or settings.data_dir / "inventory" / f"{database}.json"

    try:
        async with SessionFactory() as session:
            documents: list[DocumentInventory] = []
            for document_id in dict.fromkeys(args.document):
                document = await load_document(session, document_id)
                if document is None:
                    print(f"No document {document_id} in {database}.")
                    return 1
                documents.append(document)
            spent = await spent_today(session)
    finally:
        await engine.dispose()

    saved, before, calls = ([], [], [])
    if path.exists():
        saved, before, calls = from_json(json.loads(path.read_text()))
    # Only the documents of this run are saved, so leaving one out would drop its answers.
    if left_out := [d.document_id for d in saved if d.document_id not in args.document]:
        print(
            f"{path} also holds document(s) {', '.join(map(str, left_out))}: name them too, "
            "or keep this run apart with --out."
        )
        return 1
    by_id = {document.document_id: document for document in saved}
    for document in documents:
        earlier = by_id.get(document.document_id)
        if earlier is not None and not resume(document, earlier):
            print(f"Document {document.document_id}'s passages have changed: reading it again.")
    review = None
    if review_path(path).exists():
        review = Review.model_validate_json(review_path(path).read_text())
        try:
            reviewed(documents, review)
        except ValueError as exc:
            print(f"The review in {review_path(path)} does not fit these readings: {exc}.")
            return 1

    print(f"Inventory of {database}, saved to {path}")
    show_plan(documents)
    prompts, low, high = estimate(documents)
    print(
        f"\nStill to send: about {prompts:,} prompt tokens, plus {low:,}-{high:,} written back "
        f"(assumed), and a few hundred for comparing ideas across documents.\n"
        f"In all about {prompts + low:,}-{prompts + high:,} tokens."
    )
    for name, (requests, tokens) in spent.items():
        print(f"{name} in the last day: {requests} requests, {tokens:,} tokens (questions only).")
    if args.dry_run:
        return 0
    if args.show:
        if review is not None:
            print(f"Review: {review_path(path)}")
        save(path, documents, before, calls, review)
        show_inventory(documents, reviewed(documents, review), before)
        print(f"\nSaved to {path}")
        return 0

    if len(documents) > 1 and (problems := await needs_ollama(settings)):
        print("Comparing ideas across documents needs the embedding model:", *problems, sep="\n  ")
        return 1
    model = paced_generation_model(settings, spent, groq_only=True)
    pairs: list[Comparison] = before
    try:
        for document in documents:
            count = len(document.windows)
            for number, window in enumerate(document.windows, start=1):
                if window.reading is not None:
                    continue
                say(f"document {document.document_id}: reading window {number} of {count}")
                await read_window(model, document, number)
                save(path, documents, pairs, calls, review)
                usage = window.call.usage if window.call else {}
                say(f"  {len(window.reading.ideas) if window.reading else 0} ideas, {usage}")
            if document.needs_merge:
                say(f"document {document.document_id}: merging {len(document.readings())} readings")
                merge = await merge_document(model, document)
                save(path, documents, pairs, calls, review)
                say(f"  {len(merge.groups)} group(s)")
        concepts = reviewed(documents, review)
        if len(documents) > 1:
            async with embedding_model(settings) as embedder:
                pairs = await close_pairs(embedder, concepts)
            carry_verdicts(pairs, before)
            titles = {document.document_id: document.title for document in documents}
            say(f"comparing {sum(p.same is None for p in pairs)} close pair(s) across documents")
            calls += await compare(model, pairs, concepts, titles)
            save(path, documents, pairs, calls, review)
    except Exception as exc:
        # Everything answered so far is kept, and the same command carries on from here.
        save(path, documents, pairs, calls, review)
        if out_of_budget(exc):
            print(f"\nStopped: the day's budget is gone ({exc}). Run the same command again later.")
        else:
            print(f"\nStopped on an error: {type(exc).__name__}: {str(exc)[:500]}")
        return 1

    show_inventory(documents, concepts, pairs)
    every: list[Call] = [call for document in documents for call in document.calls] + calls
    print(f"\nSaved to {path}\nCost: {total_usage(every)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--document",
        type=int,
        action="append",
        required=True,
        metavar="ID",
        help="a document to read; give it once for each",
    )
    shown = parser.add_mutually_exclusive_group()
    shown.add_argument(
        "--dry-run", action="store_true", help="show the windows and the cost, call no model"
    )
    shown.add_argument(
        "--show",
        action="store_true",
        help="work the ideas out again from the saved answers and the review, save and show "
        "them; call no model",
    )
    parser.add_argument("--out", type=Path, metavar="PATH", help="where to keep the inventory")
    parser.add_argument("--verbose", action="store_true", help="show library log messages")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    configure_logging(arguments.verbose)
    with tracing(get_settings(), "inventory"):
        sys.exit(asyncio.run(main(arguments)))

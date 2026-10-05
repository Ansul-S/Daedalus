"""Write one question for each chosen idea of the concept inventory, evidence first.

Run from backend/:  uv run python -m scripts.write_questions --idea 4.2 4.3 7.1 [--dry-run]

The ideas are those of data/inventory/<database>.json, as the review beside it leaves them; their
passages come from the database DATABASE_URL names, and nothing is written to it. Each idea is
given a style its passages support, spread across the ideas (--style 4.14=compare chooses one).
Every question is saved to data/inventory/<database>.questions.json as it is written, with the
checks it passed and failed, so that a run that stops -- on Ctrl+C, or on Groq's day running
out -- carries on where it stopped when started again; a question whose repair Groq turns down
stands as its first answer left it. Groq alone writes. The local model reads every answer as it
comes, and one it reads as recall goes back with the code's problems; each question is then
compared with the questions already in the database, which is reported and not judged. Both need
Ollama. --out keeps a run apart; the day's spend counts every question file beside the inventory.

--dry-run calls no model: it shows the styles and what the run would cost.
--show calls no model either: it works the checks out again from the saved answers, and saves
and shows them.
"""

import argparse
import asyncio
import json
import sys
import textwrap
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage
from sqlalchemy import select
from sqlalchemy.engine import make_url

from app.core.checks import check_ollama
from app.core.config import Settings, get_settings
from app.db.models import Question
from app.db.session import SessionFactory, engine
from app.llm.embeddings import Embedder, EmbeddingError
from app.llm.models import embedding_model, helper_model, paced_generation_model
from app.llm.tracing import traced, tracing
from app.questions.batch import out_of_budget, spent_today
from app.questions.generation import Source, counted
from app.questions.inventory import (
    Call,
    Concept,
    DocumentInventory,
    Review,
    cosine,
    from_json,
    load_document,
    resume,
)
from app.questions.validation import AnswerCheck, nearest_question
from app.questions.writing import (
    INSTRUCTIONS,
    PROMPT_VERSION,
    STYLE_BRIEFS,
    Entry,
    IdeaQuestion,
    ReadingFailed,
    assign_styles,
    check_question,
    entries_json,
    misfit,
    read_entries,
    request,
    sources_of,
    supported_styles,
    write_question,
)
from scripts.ingest import configure_logging
from scripts.inventory import review_path, reviewed

# What a dry run assumes the writer sends back, which only a real run can tell: generate-v6's
# answers came to 450 to 950 tokens in one call. The prompts are counted as the pacer counts
# them, a token to four characters.
ANSWER = (600, 1_200)
# What a repair adds to the conversation it sends back
REPAIR_REQUEST = 200
# generate-v6 sent 6 of its 20 questions back for the code's problems, and the local model read
# 6 as recall, one of them among the 6: 11 of 20 would have gone back with recall sent back too.
REPAIRED = 11 / 20
# Groq's free tier, which the run waits on
TOKENS_PER_MINUTE = 8_000


def say(message: str) -> None:
    print(f"  {message}", flush=True)


def questions_path(inventory: Path) -> Path:
    """Where the questions written from the inventory at `inventory` are kept:
    data/inventory/<database>.questions.json"""
    return inventory.with_name(f"{inventory.stem}.questions.json")


def style_choice(text: str) -> tuple[str, str]:
    key, _, style = text.partition("=")
    if not key or style not in STYLE_BRIEFS:
        raise argparse.ArgumentTypeError(
            f"{text}: give IDEA=STYLE, the style one of {', '.join(STYLE_BRIEFS)}"
        )
    return key, style


def recent(calls: Iterable[Call], now: datetime) -> dict[str, tuple[int, int]]:
    """The requests and tokens of these calls in the last day, by model."""
    spent: dict[str, tuple[int, int]] = {}
    for call in calls:
        if datetime.fromisoformat(call.at) <= now - timedelta(days=1):
            continue
        requests, tokens = spent.get(call.model, (0, 0))
        used = call.usage.get("input_tokens", 0) + call.usage.get("output_tokens", 0)
        spent[call.model] = (requests + call.usage.get("requests", 0), tokens + used)
    return spent


def added(*spends: Mapping[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
    total: dict[str, tuple[int, int]] = {}
    for spend in spends:
        for model, (requests, tokens) in spend.items():
            before = total.get(model, (0, 0))
            total[model] = (before[0] + requests, before[1] + tokens)
    return total


def estimate(plans: list[tuple[Concept, list[Source], str]]) -> tuple[int, int, int, int]:
    """The prompt tokens of the first requests still to send, and the tokens of the run in all:
    with no question sent back, at generate-v6's rate of repairs with recall sent back too, and
    with every question sent back once."""
    prompts = [
        len(INSTRUCTIONS + request(concept, sources, style)) // 4
        for concept, sources, style in plans
    ]
    low, high = ANSWER
    middle = (low + high) // 2

    def total(answer: int, repaired: float) -> int:
        first = sum(prompt + answer for prompt in prompts)
        again = sum(prompt + answer + REPAIR_REQUEST + answer for prompt in prompts)
        return round(first + repaired * again)

    return sum(prompts), total(low, 0), total(middle, REPAIRED), total(high, 1)


def show_plan(
    keys: list[str],
    concepts: Mapping[str, Concept],
    styles: Mapping[str, str],
    by_key: Mapping[str, Entry],
    passages: Mapping[str, list[str]],
) -> None:
    print(f"\n  {'idea':<6} {'style':<14} {'its passages support':<54} name")
    for key in keys:
        saved = by_key[key]
        state = "  (written)" if saved.written else ""
        if not saved.written and saved.error:
            state = f"  (failed before: {saved.error[:80]})"
        supported = ", ".join(supported_styles(concepts[key], passages[key]))
        print(f"  {key:<6} {styles[key]:<14} {supported:<54} {concepts[key].name}{state}")
    counts = Counter(styles[key] for key in keys)
    print(f"  Styles: {', '.join(f'{style} {n}' for style, n in counts.most_common())}")


def show_questions(data: dict[str, Any], keys: list[str] | None = None) -> None:
    shown = [saved for saved in data["questions"] if keys is None or saved["key"] in keys]
    for saved in shown:
        if not saved["answers"]:
            print(f"\n{saved['key']} {saved['style']}: not written ({saved['error'] or 'not yet'})")
            continue
        if saved["accepted"]:
            verdict = "ACCEPTED"
        elif saved["failed"]:
            verdict = f"REJECTED: {', '.join(saved['failed'])}"
        else:
            verdict = "not checked yet"
        tries = len(saved["answers"])
        print(f"\n{saved['key']} {saved['style']} ({tries} answer{'s' * (tries > 1)}): {verdict}")
        print(textwrap.fill(saved["question"], 100, initial_indent="  ", subsequent_indent="  "))
        for problem in saved["problems"]:
            print(textwrap.shorten(f"    - {problem}", 160))
        if saved["error"]:
            print(textwrap.shorten(f"    - its repair failed: {saved['error']}", 160))
        if saved["check"] is not None and not saved["check"]["answerable"]:
            print(textwrap.shorten(f"    - not answerable: {saved['check']['missing']}", 160))
        duplicate = saved["duplicate"] or {}
        line = duplicate.get("line", 1.0)
        if (nearest := duplicate.get("database")) and nearest["similarity"] >= line:
            print(f"    - near q{nearest['question']} ({nearest['similarity']:.2f}): ", end="")
            print(textwrap.shorten(nearest["text"], 100))
        if (new := duplicate.get("new")) and new["similarity"] >= line:
            print(f"    - near the new question for {new['key']} ({new['similarity']:.2f})")
    written = [saved for saved in shown if saved["answers"]]
    failures = Counter(name for saved in written for name in saved["failed"])
    print(
        f"\nAccepted {sum(saved['accepted'] for saved in written)} of {len(written)} written; "
        f"failed: {', '.join(f'{name} {n}' for name, n in failures.most_common()) or 'none'}."
    )


def saved_entries(path: Path) -> list[Entry]:
    return read_entries(json.loads(path.read_text())) if path.exists() else []


def recorded_elsewhere(inventory: Path, path: Path) -> list[Call]:
    """The requests recorded in the other question files beside the inventory -- an earlier run,
    or one kept apart with --out -- which the day's spend counts as well as this file's."""
    files = sorted(inventory.parent.glob(f"{inventory.stem}.questions*.json"))
    return [
        call
        for other in files
        if other.resolve() != path.resolve()
        for entry in saved_entries(other)
        for call in entry.calls
    ]


def save(path: Path, entries: list[Entry], sources: Mapping[str, list[Source]]) -> dict[str, Any]:
    """Written whole to a temporary file and moved into place, so a stop mid-write never leaves
    half a file."""
    data = entries_json(entries, dict(sources))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    temporary.replace(path)
    return data


async def needs_ollama(settings: Settings) -> list[str]:
    """What stops the local models from checking and comparing the questions, if anything."""
    wanted = {"ollama", f"model {settings.helper_model}", f"model {settings.embedding_model}"}
    return [
        f"{check.name}: {check.detail}"
        for check in await check_ollama(settings)
        if check.status == "fail" and check.name in wanted
    ]


async def write(
    model: Model, checker: Model, entry: Entry, concept: Concept, sources: list[Source]
) -> None:
    """Write an entry's question, the local model reading every answer, and record what it cost
    whether or not it is written.

    A writing that fails after an answer -- its repair turned down -- keeps what it wrote, and
    the question stands as that answer left it. One the day's budget stops is written again by
    the next run, which has the budget for its repair, and so is one the local model could not
    read.
    """
    used = RunUsage()
    answers: list[IdeaQuestion] = []

    async def read(question: str) -> tuple[AnswerCheck, str]:
        return await check_question(checker, question, sources)

    try:
        with traced("generate", version=PROMPT_VERSION, idea=entry.key, style=entry.style):
            written = await write_question(
                model, concept, sources, entry.style, usage=used, answers=answers, read=read
            )
    except Exception as exc:
        entry.error = f"{type(exc).__name__}: {exc}"[:2000]
        if used.requests:
            entry.calls.append(call(model.model_name, counted(used)))
        if not out_of_budget(exc) and not isinstance(exc, ReadingFailed):
            entry.answers = answers
        raise
    entry.answers, entry.error = written.answers, None
    entry.check, entry.checker_model = written.check, written.checker_model
    entry.calls.append(call(written.model, written.usage))


def call(model: str, usage: dict[str, int]) -> Call:
    return Call(
        model=model,
        version=PROMPT_VERSION,
        usage=usage,
        at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


async def compare_questions(entries: list[Entry], embedder: Embedder, line: float) -> None:
    """Each written question's nearest accepted question in the database, and its nearest among
    the new ones, by the embedding the duplicate check uses: reported, not judged."""
    written = [entry for entry in entries if entry.written]
    if not written:
        return
    try:
        vectors = await embedder.embed_documents([entry.answers[-1].question for entry in written])
    except EmbeddingError as exc:
        for entry in written:
            entry.duplicate = {"checked": False, "why": str(exc)}
        return
    async with SessionFactory() as session:
        for index, (entry, vector) in enumerate(zip(written, vectors, strict=True)):
            report: dict[str, Any] = {"checked": True, "line": line}
            nearest = await nearest_question(session, vector)
            if nearest is not None:
                text = await session.scalar(select(Question.text).where(Question.id == nearest[0]))
                report["database"] = {
                    "question": nearest[0],
                    "similarity": round(nearest[1], 4),
                    "text": text,
                }
            others = [
                (cosine(vector, other), written[number].key)
                for number, other in enumerate(vectors)
                if number != index
            ]
            if others:
                similarity, key = max(others)
                report["new"] = {"key": key, "similarity": round(similarity, 4)}
            entry.duplicate = report


async def main(args: argparse.Namespace) -> int:
    settings = get_settings()
    if settings.environment != "local" or settings.fake_models:
        print("Questions are written locally, on real models (ENVIRONMENT=local, no FAKE_MODELS).")
        return 1
    database = make_url(settings.database_url).database
    inventory_path = settings.data_dir / "inventory" / f"{database}.json"
    path = args.out or questions_path(inventory_path)
    if not inventory_path.exists():
        print(f"No inventory at {inventory_path}: read the documents with scripts.inventory first.")
        return 1
    saved_documents, _, compare_calls = from_json(json.loads(inventory_path.read_text()))
    try:
        return await run(args, settings, inventory_path, path, saved_documents, compare_calls)
    finally:
        await engine.dispose()


async def run(
    args: argparse.Namespace,
    settings: Settings,
    inventory_path: Path,
    path: Path,
    saved_documents: list[DocumentInventory],
    compare_calls: list[Call],
) -> int:
    async with SessionFactory() as session:
        documents = []
        for saved in saved_documents:
            document = await load_document(session, saved.document_id)
            if document is None or not resume(document, saved):
                print(
                    f"Document {saved.document_id} has changed since the inventory read it: "
                    "read it again with scripts.inventory."
                )
                return 1
            if not document.read or document.needs_merge:
                print(f"The inventory of document {saved.document_id} is unfinished.")
                return 1
            documents.append(document)
        spent = await spent_today(session)
    review = None
    if review_path(inventory_path).exists():
        review = Review.model_validate_json(review_path(inventory_path).read_text())
    try:
        concepts = reviewed(documents, review)
    except ValueError as exc:
        print(f"The review in {review_path(inventory_path)} does not fit the inventory: {exc}.")
        return 1

    entries = saved_entries(path)
    by_key = {entry.key: entry for entry in entries}
    if gone := [key for key in by_key if key not in concepts]:
        print(f"{path} holds questions on ideas the inventory no longer has: {', '.join(gone)}.")
        return 1
    keys = list(dict.fromkeys(args.idea or []))
    if unknown := [key for key in keys if key not in concepts]:
        print(f"The inventory has no idea {', '.join(unknown)}.")
        return 1
    if excluded := [key for key in keys if concepts[key].excluded]:
        print(f"The review leaves idea {', '.join(excluded)} out of planning.")
        return 1
    chosen = {key: concepts[key] for key in keys}
    sources = {key: sources_of(concepts[key], documents) for key in [*by_key, *keys]}
    passages = {key: [source.text for source in sources[key]] for key in keys}
    # A question already written keeps its style, so the spread around it holds from run to run.
    kept = {key: by_key[key].style for key in keys if key in by_key and by_key[key].written}
    try:
        styles = assign_styles(chosen, kept | dict(args.style or []), passages)
    except ValueError as exc:
        print(f"No styles: {exc}.")
        return 1

    for key in keys:
        planned = Entry(
            key=key,
            name=chosen[key].name,
            document_id=chosen[key].document_id,
            style=styles[key],
            chunk_ids=[source.chunk_id for source in sources[key]],
        )
        saved = by_key.get(key)
        if saved is not None and saved.written:
            if (why := misfit(saved, planned)) is not None:
                print(f"{why}. Keep this run apart with --out, or take the idea out of {path}.")
                return 1
            continue
        if saved is not None:
            # Unwritten, it takes the plan; what its failed writing cost stays on record.
            planned.calls, planned.error = saved.calls, saved.error
            entries[entries.index(saved)] = planned
        else:
            entries.append(planned)
        by_key[key] = planned

    now = datetime.now(UTC)
    inventory_calls = [call for document in saved_documents for call in document.calls]
    day = added(
        spent,
        recent([*inventory_calls, *compare_calls], now),
        recent([call for entry in entries for call in entry.calls], now),
        recent(recorded_elsewhere(inventory_path, path), now),
    )
    if args.show:
        data = save(path, entries, sources)
        show_questions(data)
        print(f"\nSaved to {path}")
        return 0

    print(f"Questions on {len(keys)} idea(s) of {inventory_path}, saved to {path}")
    show_plan(keys, concepts, styles, by_key, passages)
    to_write = [key for key in keys if not by_key[key].written]
    to_check = [key for key in keys if by_key[key].check is None]
    if to_write:
        prompts, low, expected, high = estimate(
            [(chosen[key], sources[key], styles[key]) for key in to_write]
        )
        print(
            f"\nStill to write: {len(to_write)} question(s), {prompts:,} prompt tokens to start "
            f"with.\nIn all about {low:,} tokens if none is sent back, {high:,} if every one "
            f"is sent back once, and {expected:,} at generate-v6's rate with recall sent back "
            f"(11 of 20): about {expected // TOKENS_PER_MINUTE} minutes at Groq's 8,000 tokens "
            "a minute."
        )
    print(f"Still to check with {settings.helper_model}: {len(to_check)} question(s).")
    for name, (requests, tokens) in day.items():
        print(f"{name} in the last day: {requests} requests, {tokens:,} tokens (as recorded).")
    if args.dry_run:
        return 0
    if not to_check and all(by_key[key].duplicate for key in keys):
        print("Every question is written, checked and compared.")
        return 0

    if problems := await needs_ollama(settings):
        print("Checking the questions needs the local models:", *problems, sep="\n  ")
        return 1
    model = paced_generation_model(settings, day, groq_only=True)
    checker = helper_model(settings)
    stopped = None
    for number, key in enumerate(keys, start=1):
        entry = by_key[key]
        if entry.check is not None:
            continue
        if not entry.written:
            say(f"{number}/{len(keys)} idea {key}, {entry.style}: writing")
            try:
                await write(model, checker, entry, chosen[key], sources[key])
            except Exception as exc:
                save(path, entries, sources)
                if out_of_budget(exc):
                    stopped = f"the day's budget is gone ({exc})"
                    break
                if isinstance(exc, ReadingFailed):
                    print(f"\nStopped: the local checker failed ({exc}).")
                    return 1
                say(f"  failed: {entry.error}")
                if not entry.written:
                    continue
            save(path, entries, sources)
            say(f"  {len(entry.answers)} answer(s), {entry.calls[-1].usage}")
        if entry.check is not None:
            continue
        # Kept from a writing that failed, or written before the local model read as it went
        try:
            entry.check, entry.checker_model = await check_question(
                checker, entry.answers[-1].question, sources[key]
            )
        except Exception as exc:
            save(path, entries, sources)
            print(f"\nStopped: the local checker failed ({type(exc).__name__}: {exc}).")
            return 1
        save(path, entries, sources)

    say("comparing the questions with those in the database and with each other")
    async with embedding_model(settings) as embedder:
        await compare_questions(entries, embedder, settings.duplicate_similarity)
    data = save(path, entries, sources)
    show_questions(data, keys)
    print(f"\nSaved to {path}\nWhat writing every question in it cost: {data['usage']}")
    if stopped:
        print(f"Stopped early: {stopped}. Run the same command again later.")
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--idea",
        action="extend",
        nargs="+",
        metavar="KEY",
        help='the ideas to write questions on, by their inventory keys ("4.2")',
    )
    parser.add_argument(
        "--style",
        type=style_choice,
        action="append",
        metavar="KEY=STYLE",
        help="the style for one idea, instead of the one assigned; one its passages support",
    )
    shown = parser.add_mutually_exclusive_group()
    shown.add_argument(
        "--dry-run", action="store_true", help="show the styles and the cost, call no model"
    )
    shown.add_argument(
        "--show",
        action="store_true",
        help="work the checks out again from the saved answers, save and show them; call no model",
    )
    parser.add_argument("--out", type=Path, metavar="PATH", help="where to keep the questions")
    parser.add_argument("--verbose", action="store_true", help="show library log messages")
    args = parser.parse_args()
    if not args.idea and not args.show:
        parser.error("name the ideas with --idea, or --show the saved questions")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    configure_logging(arguments.verbose)
    with tracing(get_settings(), "write_questions"):
        sys.exit(asyncio.run(main(arguments)))

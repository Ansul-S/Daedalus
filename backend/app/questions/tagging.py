"""Concept tags for every chunk, read by the small local model.

Tagging is the first half of the topic map: for each chunk the model says what the passage
explains, names the concepts in it, and judges whether a question could be asked from it
alone. Grouping those tags into topics is in `topics.py`.

The model runs with thinking off, temperature 0 and a fixed seed, so tagging the same chunk
twice gives the same tags. Its judgement is only half of `worth_asking`: a passage that is
only code, or only course scaffolding, is context for other questions whatever it says.
"""

import logging
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models import Model
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import Chunk, ChunkTags, Document
from app.ingest.chunking import SECTION_LIST_SEPARATOR
from app.llm.models import helper_settings

log = logging.getLogger(__name__)

PROMPT_VERSION = "tags-v2"
# The model is asked for this many; the ceiling is applied afterwards rather than in the
# schema, so that a sixth tag costs a trim instead of another call to a slow local model.
MAX_TAGS = 5

INSTRUCTIONS = """\
You tag passages from machine-learning study material: lecture notebooks, course notes and
research papers. Report three things about the passage you are given.

explains: what the passage teaches, as a short phrase rather than a sentence. Write exactly
"nothing" when it only sets things up -- a roadmap, a list of learning objectives, an
installation or import cell, acknowledgements, references, or a title page.

tags: 2 to 5 short concept names in lower case, such as "self attention", "positional
encoding", "vanishing gradient" or "faiss index". Name only concepts the passage itself
discusses, using the words it uses. Never add a concept that is merely related to the
subject, and never one the passage only mentions in passing.

worth_asking: true when an interview question could be answered from this passage alone,
false when the passage is scaffolding, code without explanation, a roadmap, a list of
objectives, acknowledgements or references.
"""

PASSAGE = """\
Document: {title}
Section: {section}
Passage:
<<<
{text}
>>>
"""

# Sections that exist to organize the material rather than teach it
BOILERPLATE = re.compile(
    r"\b(roadmap|learning objectives?|what you will learn|how this notebook"
    r"|prerequisites?|setup|installation|imports?|dependencies|requirements"
    r"|acknowledg\w+|references|bibliography|table of contents)\b",
    re.IGNORECASE,
)

Reporter = Callable[[str], None]


class TagReading(BaseModel):
    """What the model reports about one passage."""

    explains: str = Field(description='What the passage teaches, or "nothing"')
    tags: list[str] = Field(description="2 to 5 short concept names, lower case")
    worth_asking: bool = Field(
        description="Whether a question could be answered from this passage alone"
    )


@dataclass
class TaggedChunk:
    chunk_id: int
    explains: str
    tags: list[str]
    # The model's judgement and the rules together
    worth_asking: bool
    model_worth_asking: bool
    # Which rule made the chunk context only, if any
    skip_reason: str | None


def normalize_tag(tag: str) -> str:
    """Lower case, with hyphen and underscore variants of a name folded together, so that
    "dot-product attention" and "dot product attention" do not become two topics."""
    folded = unicodedata.normalize("NFKC", tag).lower()
    folded = re.sub(r"[-_/]+", " ", folded)
    return re.sub(r"\s+", " ", folded).strip(" .,:;\"'")


def clean_tags(tags: list[str]) -> list[str]:
    """Normalized, in the order given, without repeats, at most MAX_TAGS."""
    cleaned = dict.fromkeys(filter(None, (normalize_tag(tag) for tag in tags)))
    return list(cleaned)[:MAX_TAGS]


def context_only(chunk: Chunk) -> str | None:
    """Why a chunk can only be context for a question, or None when it can carry one.

    The model caught 2 of 5 scaffolding passages on its own in the trial, so these rules run
    beside it and a chunk has to pass both.
    """
    if "text" not in chunk.content_types:
        return "code only"
    if chunk.section is None:
        return "front matter" if chunk.position == 0 else None
    # A chunk can cover several sections, and it is only scaffolding when every one of them
    # is: the chunk that ends a paper's conclusion and runs into its acknowledgements still
    # holds the conclusion.
    sections = chunk.section.split(SECTION_LIST_SEPARATOR)
    found = [BOILERPLATE.search(section) for section in sections]
    return f"boilerplate: {found[0].group(0).lower()}" if all(found) else None


def tagger(model: Model) -> Agent[None, TagReading]:
    return Agent(model, output_type=NativeOutput(TagReading), instructions=INSTRUCTIONS)


def passage(title: str, chunk: Chunk) -> str:
    return PASSAGE.format(title=title, section=chunk.section or "-", text=chunk.text)


async def read_chunk(agent: Agent[None, TagReading], title: str, chunk: Chunk) -> TaggedChunk:
    result = await agent.run(passage(title, chunk), model_settings=helper_settings())
    reading = result.output
    reason = context_only(chunk)
    return TaggedChunk(
        chunk_id=chunk.id,
        explains=reading.explains.strip(),
        tags=clean_tags(reading.tags),
        worth_asking=reading.worth_asking and reason is None,
        model_worth_asking=reading.worth_asking,
        skip_reason=reason,
    )


async def chunks_to_tag(
    session: AsyncSession, *, document_id: int | None = None, limit: int | None = None, retag: bool
) -> list[tuple[str, Chunk]]:
    """Current chunks that need tags, each with its document title, in reading order."""
    query = (
        select(Document.title, Chunk)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.superseded_at.is_(None))
        .order_by(Chunk.document_id, Chunk.position)
    )
    if document_id is not None:
        query = query.where(Chunk.document_id == document_id)
    if not retag:
        already = select(ChunkTags.chunk_id).where(ChunkTags.chunk_id == Chunk.id)
        query = query.where(~already.exists())
    rows = await session.execute(query.limit(limit))
    return [(title, chunk) for title, chunk in rows.tuples()]


async def store(session: AsyncSession, tagged: TaggedChunk, model_name: str) -> None:
    values = {
        "chunk_id": tagged.chunk_id,
        "explains": tagged.explains,
        "tags": tagged.tags,
        "worth_asking": tagged.worth_asking,
        "model_worth_asking": tagged.model_worth_asking,
        "skip_reason": tagged.skip_reason,
        "model": model_name,
        "prompt_version": PROMPT_VERSION,
    }
    statement = insert(ChunkTags).values(values)
    await session.execute(
        statement.on_conflict_do_update(
            index_elements=[ChunkTags.chunk_id],
            set_={name: values[name] for name in values if name != "chunk_id"},
        )
    )


async def tag_chunks(
    sessions: async_sessionmaker[AsyncSession],
    model: Model,
    *,
    document_id: int | None = None,
    limit: int | None = None,
    retag: bool = False,
    report: Reporter | None = None,
) -> list[TaggedChunk]:
    """Tag every current chunk that has no tags yet, one call at a time.

    Each chunk is committed on its own: the local model takes seconds per chunk, and an
    interrupted run should not have to start over.
    """
    say = report or log.info
    agent = tagger(model)
    async with sessions() as session:
        pending = await chunks_to_tag(session, document_id=document_id, limit=limit, retag=retag)
    if not pending:
        say("every chunk is already tagged")
        return []

    tagged: list[TaggedChunk] = []
    for number, (title, chunk) in enumerate(pending, start=1):
        reading = await read_chunk(agent, title, chunk)
        async with sessions() as session, session.begin():
            await store(session, reading, model.model_name)
        tagged.append(reading)
        say(
            f"tagged {number}/{len(pending)}: {chunk.section or title} -> {', '.join(reading.tags)}"
        )
    return tagged


async def apply_rules(
    sessions: async_sessionmaker[AsyncSession], report: Reporter | None = None
) -> list[TaggedChunk]:
    """Judge the stored tags by the rules again, keeping what the model said.

    The rules change far more often than the prompt does, and re-reading the whole library
    costs the local model twenty minutes, so they are applied again on their own.
    """
    say = report or log.info
    changed: list[TaggedChunk] = []
    async with sessions() as session, session.begin():
        rows = await session.execute(
            select(Chunk, ChunkTags).join(ChunkTags, ChunkTags.chunk_id == Chunk.id)
        )
        for chunk, tags in rows.tuples():
            reason = context_only(chunk)
            worth_asking = tags.model_worth_asking and reason is None
            if (reason, worth_asking) == (tags.skip_reason, tags.worth_asking):
                continue
            asked = "asked about"
            say(f"chunk {chunk.id}: {tags.skip_reason or asked} -> {reason or asked}")
            tags.skip_reason = reason
            tags.worth_asking = worth_asking
            changed.append(
                TaggedChunk(
                    chunk_id=chunk.id,
                    explains=tags.explains,
                    tags=tags.tags,
                    worth_asking=worth_asking,
                    model_worth_asking=tags.model_worth_asking,
                    skip_reason=reason,
                )
            )
    say(f"{len(changed)} chunks changed")
    return changed

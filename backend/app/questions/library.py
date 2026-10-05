"""Questions into the library: a written question as the library keeps it, and corrections made
to one by hand, kept in its report.

The writer (`app.questions.writing`) saves its questions to a file, and a person's review decides
which of them go into the library: each one as written, or with the edit that fixes it. Only a
question that passed every check can go in. Stored, it carries a report the question page reads
-- its key points' quotes, one question asked, the local model's reading, the nearest question --
and beside them the idea it asks about, every sentence its key points could rest on, and the
review that kept it.

A question in the library can be corrected, under the rule generation works to: every key
point's quote has to be in the passage it names. Each change is written into the question's
report next to the checks it first went through, with what it replaced, so the page that shows
the question shows the correction too.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select, union
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Attempt,
    Card,
    ChunkTopic,
    Question,
    QuestionSource,
    Rating,
    Review,
    Topic,
)
from app.llm.embeddings import Embedder
from app.questions.generation import KeyPoint, Source
from app.questions.grounding import QuoteCheck, check_quote
from app.questions.inventory import total_usage
from app.questions.topics import topic_for
from app.questions.validation import nearest_question
from app.questions.writing import (
    MISCONCEPTIONS,
    Entry,
    check_answer,
    checker_failed,
    library_key_points,
)


def quote_report(check: QuoteCheck) -> dict[str, Any]:
    """A quote as a question's report keeps it."""
    return {
        "quote": check.quote,
        "chunk_id": check.chunk_id,
        "score": round(check.score, 1),
        "problem": check.problem,
    }


def correct(
    question: Question,
    proposed: Mapping[str, Any],
    *,
    reason: str | None,
    quotes: Sequence[QuoteCheck] | None = None,
    vector: list[float] | None = None,
) -> dict[str, Any] | None:
    """Make the changes `proposed` names and keep them in the question's report, each with what
    it replaced. A field given as None stays as it is, and a change to what the question already
    says is no change at all. `quotes` are the new key points' quotes, checked against their
    passages; `vector` is the new text's embedding, None while the embedding model is away, which
    clears the old one. Returns the edit as kept, or None when nothing changed."""
    changes = {
        name: {"from": getattr(question, name), "to": value}
        for name, value in proposed.items()
        if value is not None and value != getattr(question, name)
    }
    if not changes:
        return None
    edit: dict[str, Any] = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "reason": reason or None,
        "changes": changes,
    }
    for name, change in changes.items():
        setattr(question, name, change["to"])
    if "key_points" in changes and quotes is not None:
        edit["quotes"] = [quote_report(check) for check in quotes]
    if "text" in changes:
        question.embedding = vector
        edit["embedding"] = "updated" if vector is not None else "cleared"
    # A new dict, since the column does not track changes made inside the old one
    question.validation = question.validation | {
        "edits": [*question.validation.get("edits", []), edit]
    }
    return edit


class Verdict(BaseModel):
    """The review's reading of a question it keeps."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["good", "fix"]
    reason: str = Field(min_length=1)
    # When it was given
    at: str


class Correction(BaseModel):
    """The edit that fixes a question before it goes in, made as the question bank makes one: a
    field left out stays as it is, and the key points are replaced as a whole."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1)
    text: str | None = Field(None, min_length=1)
    reference_answer: str | None = Field(None, min_length=1)
    key_points: list[KeyPoint] | None = Field(None, min_length=2, max_length=4)

    @model_validator(mode="after")
    def changes_something(self) -> Self:
        if self.text is None and self.reference_answer is None and self.key_points is None:
            raise ValueError("an edit changes the text, the reference answer or the key points")
        return self


class Kept(BaseModel):
    """A question the review keeps: the idea it asks about, in the questions file beside the
    inventory it was written into. A fixable one goes in only with the edit that fixes it."""

    model_config = ConfigDict(extra="forbid")

    file: str = Field(pattern=r"^[^/\\]+\.json$")
    key: str
    review: Verdict
    edit: Correction | None = None
    # The topic to file it under when the one its passages share most is not what it asks about:
    # one of its passages' topics, by name
    topic: str | None = Field(None, min_length=1)

    @model_validator(mode="after")
    def fixed(self) -> Self:
        if self.review.verdict == "fix" and self.edit is None:
            raise ValueError(f"{self.key} is fixable, so it goes in only with the edit fixing it")
        return self


class Selection(BaseModel):
    """What a review keeps for the library, in the order it goes in: each idea once."""

    model_config = ConfigDict(extra="forbid")

    questions: list[Kept]

    @model_validator(mode="after")
    def once(self) -> Self:
        counts = Counter(kept.key for kept in self.questions)
        if twice := sorted(key for key, count in counts.items() if count > 1):
            raise ValueError(f"idea {', '.join(twice)} is kept more than once")
        return self


def unfit(entry: Entry, sources: Sequence[Source]) -> str | None:
    """Why a written question cannot go into the library, or None when it can: it has to be
    written from the passages its idea has now, read by the local model, and pass every check."""
    if not entry.written:
        return "it is not written"
    if entry.chunk_ids != [source.chunk_id for source in sources]:
        return "its idea's passages are not the ones it was written from"
    if entry.check is None:
        return "the local model has not read it"
    failed = check_answer(entry.answers[-1], sources).failed + checker_failed(entry.check)
    if failed:
        return f"it failed {', '.join(failed)}"
    return None


def library_question(entry: Entry, sources: Sequence[Source], kept: Kept) -> Question:
    """A written question as the library keeps it, accepted, with the report the question page
    reads -- each key point's quote, one question asked, the local model's reading -- and beside
    it the idea, every evidence sentence as checked, and the review. `unfit` comes first."""
    answer = entry.answers[-1]
    checks = check_answer(answer, sources)
    reading = entry.check
    assert reading is not None, "a question goes in only once the local model has read it"
    report: dict[str, Any] = {
        "prompt_version": entry.version,
        "quotes": [
            quote_report(checks.evidence[point.evidence - 1]) for point in answer.key_points
        ],
        "compound": checks.compound,
        "kind": reading.kind,
        "answerable": reading.answerable,
        "missing": reading.missing.strip(),
        "checker_answer": reading.answer.strip(),
        "checker_model": entry.checker_model,
        "failed": [],
        "idea": {"key": entry.key, "name": entry.name, "document_id": entry.document_id},
        "evidence": [quote_report(check) for check in checks.evidence],
        "written": {"file": kept.file, "answers": len(entry.answers)},
        "review": kept.review.model_dump() | ({"topic": kept.topic} if kept.topic else {}),
    }
    return Question(
        text=answer.question,
        reference_answer=answer.reference_answer,
        key_points=library_key_points(answer, checks),
        misconceptions=answer.misconceptions[:MISCONCEPTIONS],
        style=entry.style,
        difficulty=answer.difficulty,
        status="accepted",
        validation=report,
        generator_model=entry.calls[-1].model,
        prompt_version=entry.version,
        usage=total_usage(entry.calls) | {"attempts": len(entry.answers)},
    )


def edit_quotes(edit: Correction, sources: Sequence[Source]) -> list[QuoteCheck] | None:
    """The edit's key point quotes, each checked against the passage it names; None when the
    edit leaves the key points as they are."""
    if edit.key_points is None:
        return None
    texts = {source.chunk_id: source.text for source in sources}
    return [
        check_quote(point.evidence_quote, point.chunk_id, texts.get(point.chunk_id))
        for point in edit.key_points
    ]


async def passage_topic(session: AsyncSession, chunk_ids: Sequence[int], name: str) -> int:
    """The topic of that name, when one of these passages is under it; a question is filed only
    under a topic its passages carry."""
    found = await session.scalar(
        select(Topic.id)
        .join(ChunkTopic, ChunkTopic.topic_id == Topic.id)
        .where(ChunkTopic.chunk_id.in_(chunk_ids), Topic.name == name)
        .limit(1)
    )
    if found is None:
        raise ValueError(f"no passage of it is under the topic {name!r}")
    return found


async def store(
    session: AsyncSession,
    entry: Entry,
    sources: Sequence[Source],
    kept: Kept,
    embedder: Embedder,
) -> Question:
    """Put a kept question into the library: its passages as its sources, filed under the topic
    they share most or the one the review names, embedded for the duplicate check, with its
    nearest question and how far its reference answer drifts from the local model's recorded as
    validation records them. Then its edit, if it has one. An edit quoting what is not in the
    passage it names is refused, and nothing of the question is kept."""
    question = library_question(entry, sources, kept)
    quotes = edit_quotes(kept.edit, sources) if kept.edit is not None else None
    if quotes is not None and (missed := [check for check in quotes if not check.grounded]):
        problems = "; ".join(f'"{check.quote}": {check.problem}' for check in missed)
        raise ValueError(
            f"idea {entry.key}: the edit's key points quote what is not there: {problems}"
        )
    assert entry.check is not None
    # One request for all three, as validation makes it
    embedding, reference, checked = await embedder.embed_documents(
        [question.text, question.reference_answer, entry.check.answer]
    )
    nearest = await nearest_question(session, embedding)
    question.validation = question.validation | {
        "nearest_question": nearest[0] if nearest else None,
        "nearest_similarity": round(nearest[1], 4) if nearest else None,
        "answer_agreement": round(sum(a * b for a, b in zip(reference, checked, strict=True)), 4),
    }
    question.embedding = embedding
    question.topic_id = (
        await topic_for(session, entry.chunk_ids)
        if kept.topic is None
        else await passage_topic(session, entry.chunk_ids, kept.topic)
    )
    session.add(question)
    await session.flush()
    session.add_all(
        QuestionSource(question_id=question.id, chunk_id=chunk_id, position=position)
        for position, chunk_id in enumerate(entry.chunk_ids)
    )
    if kept.edit is not None:
        edit = kept.edit
        vector = None
        if edit.text is not None and edit.text != question.text:
            [vector] = await embedder.embed_documents([edit.text])
        proposed = {
            "text": edit.text,
            "reference_answer": edit.reference_answer,
            "key_points": None
            if edit.key_points is None
            else [point.model_dump() for point in edit.key_points],
        }
        correct(question, proposed, reason=edit.reason, quotes=quotes, vector=vector)
    return question


async def in_library(session: AsyncSession) -> dict[str, int]:
    """The questions a review put in, by the idea they ask about."""
    rows = await session.execute(
        select(Question.validation["idea"]["key"].astext, Question.id).where(
            Question.validation.has_key("review")
        )
    )
    return {key: question_id for key, question_id in rows.all()}


async def earlier(session: AsyncSession) -> list[int]:
    """The questions no review put in: those of a library written before this one."""
    rows = await session.scalars(
        select(Question.id).where(~Question.validation.has_key("review")).order_by(Question.id)
    )
    return list(rows)


async def practised(session: AsyncSession, question_ids: Sequence[int]) -> list[int]:
    """Those of these questions someone has answered, scheduled, reviewed or rated."""
    if not question_ids:
        return []
    used = union(
        *(
            select(table.question_id).where(table.question_id.in_(question_ids))
            for table in (Attempt, Card, Review, Rating)
        )
    )
    return sorted(await session.scalars(select(used.subquery().c.question_id)))

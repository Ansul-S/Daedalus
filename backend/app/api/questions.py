"""Question endpoints: start a batch, read what came out of it, correct or retire a question,
and browse the topic map or have it built.

Starting a batch only writes down the plan; `make worker` (or `make generate`) does the
writing. Checking a question and testing it for duplicates both run on the local models, so
starting a batch is a local-only endpoint, like adding material. So is building the topic
map, which the worker does with the small local model.

A question is shown with everything that stands behind it: the chunks it was written from,
cited the same way search results are, the key points an answer has to cover with the quote
that proves each one, and the report from every check it went through, whether it passed or
failed. It also comes with your latest rating of it (`app.api.ratings`), and the list can be
narrowed down by that rating.

A question in the library can be corrected by hand, under the rule generation works to: every
key point's quote has to be in the passage it names. It can also be retired, which takes it
out of practice and out of the duplicate check without deleting it or its answers. Each change
is written into the question's report next to the checks it first went through.
"""

import logging
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, Self, get_args

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import ColumnElement, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.documents import JobOut
from app.api.ratings import (
    RATING_VALUES,
    RatingFilter,
    RatingOut,
    latest_rating,
    question_ratings,
)
from app.api.search import citation, get_embedder, source_link
from app.core.config import Settings, get_settings
from app.db.models import (
    QUESTION_STATUSES,
    Chunk,
    ChunkTags,
    ChunkTopic,
    Document,
    Job,
    Question,
    QuestionSource,
    Topic,
    source_updated,
)
from app.db.session import get_session
from app.ingest import queue
from app.llm.embeddings import Embedder, EmbeddingError
from app.questions.batch import start_run
from app.questions.generation import Style
from app.questions.grounding import QuoteCheck, check_quote

log = logging.getLogger(__name__)

router = APIRouter(tags=["questions"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
EmbedderDep = Annotated[Embedder | None, Depends(get_embedder)]

Status = Literal["accepted", "rejected", "retired"]
assert set(get_args(Status)) == set(QUESTION_STATUSES)

# Room to spare over the longest the generator has written (a 319-character question, a
# 935-character reference answer, a 185-character key point, a 468-character quote), short of
# pasting in a passage
MAX_TEXT_CHARS = 1000
MAX_REFERENCE_CHARS = 3000
MAX_POINT_CHARS = 500
MAX_QUOTE_CHARS = 1000
MAX_REASON_CHARS = 500
# The fields an edit can change
EDITABLE = ("text", "reference_answer", "key_points", "status")


def local_only(settings: SettingsDep) -> None:
    if settings.environment != "local":
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Writing questions runs locally; start a batch from your own machine",
        )


def local_map_only(settings: SettingsDep) -> None:
    if settings.environment != "local":
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "The topic map is built locally; build it from your own machine",
        )


class GenerateIn(BaseModel):
    count: int = Field(10, ge=1, le=100, description="How many questions to write")
    document_id: int | None = Field(None, description="Ask about one document only")


class GenerateOut(BaseModel):
    message: str
    # How many questions the job set out to write, which is less than asked for once the
    # library runs out of passages nothing has been asked about yet
    planned: int
    job: JobOut | None


class KeyPointOut(BaseModel):
    text: str
    weight: int
    evidence_quote: str
    chunk_id: int | None = None


class KeyPointIn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    text: str = Field(min_length=1, max_length=MAX_POINT_CHARS)
    weight: Literal[1, 2, 3] = Field(description="How much of the answer the point carries")
    evidence_quote: str = Field(
        min_length=1,
        max_length=MAX_QUOTE_CHARS,
        description="Six or more words copied from the passage, showing the point is there",
    )
    chunk_id: int = Field(description="The passage the quote is from: one of the sources")


class QuestionEditIn(BaseModel):
    """The changes to make to a question. A field left out stays as it is."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    text: str | None = Field(None, min_length=1, max_length=MAX_TEXT_CHARS)
    reference_answer: str | None = Field(None, min_length=1, max_length=MAX_REFERENCE_CHARS)
    # Replaced as a whole, in the order given: two to four, as generation writes them
    key_points: list[KeyPointIn] | None = Field(None, min_length=2, max_length=4)
    # In or out of the library; a rejected question stays rejected
    status: Literal["accepted", "retired"] | None = None
    # Why, kept with the change
    reason: str | None = Field(None, max_length=MAX_REASON_CHARS)

    @model_validator(mode="after")
    def changes_something(self) -> Self:
        if all(getattr(self, name) is None for name in EDITABLE):
            raise ValueError(f"nothing to change: give any of {', '.join(EDITABLE)}")
        return self


class SourceOut(BaseModel):
    chunk_id: int
    document_id: int
    document_title: str
    citation: str
    link: str | None
    section: str | None
    # Whether a later ingestion has replaced this passage
    superseded: bool
    text: str


class QuestionOut(BaseModel):
    id: int
    text: str
    style: str
    difficulty: int
    status: str
    topic_id: int | None
    topic: str | None
    # True when any source has been replaced by a later ingestion: the question still points
    # at the text it was written from, but the document has moved on
    source_updated: bool
    citations: list[str]
    document_ids: list[int]
    created_at: datetime
    # Your latest rating of the question; null until it is rated
    rating: RatingOut | None


class QuestionDetailOut(QuestionOut):
    reference_answer: str
    key_points: list[KeyPointOut]
    misconceptions: list[str]
    # What each check found, whether or not the question passed
    validation: dict[str, Any]
    generator_model: str
    prompt_version: str
    usage: dict[str, Any]
    sources: list[SourceOut]


class QuestionsOut(BaseModel):
    total: int
    limit: int
    offset: int
    results: list[QuestionOut]


class TopicOut(BaseModel):
    id: int
    name: str
    tags: list[str]
    # Current chunks filed under the topic, and the questions written from them
    chunk_count: int
    question_count: int
    accepted_count: int


class TopicMapOut(BaseModel):
    message: str
    # Current chunks the local model has not tagged yet: the job tags them before it groups
    # every tag in the library into topics
    untagged: int
    job: JobOut | None


def _filters(
    status: Status | None,
    topic_id: int | None,
    style: Style | None,
    difficulty: int | None,
    document_id: int | None,
    updated: bool | None,
    rating: RatingFilter | None,
) -> list[ColumnElement[bool]]:
    """The conditions behind a listing, shared by the page and its total."""
    conditions: list[ColumnElement[bool]] = []
    if status is not None:
        conditions.append(Question.status == status)
    if topic_id is not None:
        conditions.append(Question.topic_id == topic_id)
    if style is not None:
        conditions.append(Question.style == style)
    if difficulty is not None:
        conditions.append(Question.difficulty == difficulty)
    if document_id is not None:
        conditions.append(
            Question.id.in_(
                select(QuestionSource.question_id)
                .join(Chunk, Chunk.id == QuestionSource.chunk_id)
                .where(Chunk.document_id == document_id)
            )
        )
    if updated is not None:
        conditions.append(source_updated() if updated else ~source_updated())
    if rating == "unrated":
        conditions.append(latest_rating().is_(None))
    elif rating is not None:
        conditions.append(latest_rating() == RATING_VALUES[rating])
    return conditions


async def _sources(
    session: AsyncSession, question_ids: list[int]
) -> dict[int, list[tuple[Document, Chunk]]]:
    """Every question's sources in one query, in the order the model was shown them."""
    if not question_ids:
        return {}
    rows = await session.execute(
        select(QuestionSource.question_id, Document, Chunk)
        .join(Chunk, Chunk.id == QuestionSource.chunk_id)
        .join(Document, Document.id == Chunk.document_id)
        .where(QuestionSource.question_id.in_(question_ids))
        .order_by(QuestionSource.question_id, QuestionSource.position)
    )
    found: dict[int, list[tuple[Document, Chunk]]] = {}
    for question_id, document, chunk in rows.tuples():
        found.setdefault(question_id, []).append((document, chunk))
    return found


def _question_out(
    question: Question,
    topic: str | None,
    updated: bool,
    sources: list[tuple[Document, Chunk]],
    rating: RatingOut | None,
) -> QuestionOut:
    return QuestionOut(
        id=question.id,
        text=question.text,
        style=question.style,
        difficulty=question.difficulty,
        status=question.status,
        topic_id=question.topic_id,
        topic=topic,
        source_updated=updated,
        citations=[citation(document, chunk) for document, chunk in sources],
        document_ids=list(dict.fromkeys(document.id for document, _ in sources)),
        created_at=question.created_at,
        rating=rating,
    )


@router.post(
    "/questions/generate",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(local_only)],
)
async def generate_questions(
    body: GenerateIn, session: SessionDep, response: Response
) -> GenerateOut:
    """Plan a batch of questions and queue it. Returns 202 with the job the worker will run.

    One batch at a time: while a job is still queued or running it is returned unchanged,
    since a second plan made now would pick the same passages and pay for them twice.
    """
    if body.document_id is not None and await session.get(Document, body.document_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    active = await session.scalar(
        select(Job)
        .where(Job.kind == "generate", Job.status.in_(queue.ACTIVE_STATUSES))
        .order_by(Job.id.desc())
        .limit(1)
    )
    if active is not None:
        return GenerateOut(
            message=f"already {active.status}",
            planned=active.options.get("planned", 0),
            job=JobOut.model_validate(active),
        )

    started = await start_run(session, body.count, document_id=body.document_id)
    if started is None:
        # Nothing to do, so nothing is written down: the library holds no passage that is
        # worth asking about and not already covered by an accepted question.
        response.status_code = status.HTTP_200_OK
        return GenerateOut(
            message="nothing left to ask about; add material, or build the topic map first",
            planned=0,
            job=None,
        )
    job, tasks = started
    await session.commit()
    await session.refresh(job)
    return GenerateOut(message="queued", planned=len(tasks), job=JobOut.model_validate(job))


@router.get("/questions")
async def list_questions(
    session: SessionDep,
    status: Annotated[Status | None, Query(description="Only questions in this state")] = None,
    topic_id: Annotated[int | None, Query(description="Only questions under this topic")] = None,
    style: Annotated[Style | None, Query(description="Only questions of this shape")] = None,
    difficulty: Annotated[int | None, Query(ge=1, le=5)] = None,
    document_id: Annotated[int | None, Query(description="Only questions from this source")] = None,
    updated: Annotated[
        bool | None,
        Query(
            alias="source_updated",
            description="Only questions whose sources a later ingestion replaced",
        ),
    ] = None,
    rating: Annotated[
        RatingFilter | None,
        Query(description="Only questions whose latest rating is good or poor, or never rated"),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> QuestionsOut:
    """The questions in the library, newest first. Every filter is optional."""
    conditions = _filters(status, topic_id, style, difficulty, document_id, updated, rating)
    total = await session.scalar(select(func.count()).select_from(Question).where(*conditions))
    rows = await session.execute(
        select(Question, Topic.name, source_updated().label("updated"))
        .outerjoin(Topic, Topic.id == Question.topic_id)
        .where(*conditions)
        .order_by(Question.created_at.desc(), Question.id.desc())
        .limit(limit)
        .offset(offset)
    )
    found = rows.all()
    question_ids = [question.id for question, _, _ in found]
    sources = await _sources(session, question_ids)
    ratings = await question_ratings(session, question_ids)
    return QuestionsOut(
        total=total or 0,
        limit=limit,
        offset=offset,
        results=[
            _question_out(
                question, topic, updated, sources.get(question.id, []), ratings.get(question.id)
            )
            for question, topic, updated in found
        ],
    )


@router.get("/questions/{question_id}")
async def get_question(question_id: int, session: SessionDep) -> QuestionDetailOut:
    """One question with its sources, key points and validation report."""
    return await _detail(session, question_id)


@router.patch("/questions/{question_id}")
async def edit_question(
    question_id: int, body: QuestionEditIn, session: SessionDep, embedder: EmbedderDep
) -> QuestionDetailOut:
    """Correct a question, or retire it from the library and put it back.

    Key points are replaced as a whole, and every quote has to be in the passage it names, as
    in generation; one that is not is refused with 422 and nothing changes. A new question
    text is embedded again for the duplicate check. While the local embedding model is away
    the old embedding is cleared instead, so the check passes over the question rather than
    compare against words it no longer has. Each change is kept in the question's report with
    what it replaced; a change to what the question already says is no change at all.
    """
    question = await session.get(Question, question_id)
    if question is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "question not found")
    if question.status == "rejected":
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "a rejected question is kept as the record of why it was turned down; "
            "it can't be edited or put in the library",
        )
    checks = await _ground(session, question_id, body.key_points) if body.key_points else None
    vector = None
    if body.text is not None and body.text != question.text:
        vector = await _embed(embedder, body.text)

    # Locked only after the embedding model has answered, so a slow model holds up nothing.
    # The row is read again under the lock: what the edit replaces is what is there now.
    question = await session.get_one(
        Question, question_id, with_for_update={"key_share": True}, populate_existing=True
    )
    proposed: dict[str, Any] = {
        "text": body.text,
        "reference_answer": body.reference_answer,
        "key_points": None
        if body.key_points is None
        else [point.model_dump() for point in body.key_points],
        "status": body.status,
    }
    changes = {
        name: {"from": getattr(question, name), "to": value}
        for name, value in proposed.items()
        if value is not None and value != getattr(question, name)
    }
    if changes:
        edit: dict[str, Any] = {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "reason": body.reason or None,
            "changes": changes,
        }
        for name, change in changes.items():
            setattr(question, name, change["to"])
        if "key_points" in changes and checks is not None:
            edit["quotes"] = [
                {
                    "quote": check.quote,
                    "chunk_id": check.chunk_id,
                    "score": round(check.score, 1),
                    "problem": check.problem,
                }
                for check in checks
            ]
        if "text" in changes:
            question.embedding = vector
            edit["embedding"] = "updated" if vector is not None else "cleared"
        # A new dict, since the column does not track changes made inside the old one
        question.validation = question.validation | {
            "edits": [*question.validation.get("edits", []), edit]
        }
    await session.commit()
    return await _detail(session, question_id)


async def _ground(
    session: AsyncSession, question_id: int, points: list[KeyPointIn]
) -> list[QuoteCheck]:
    """Check each key point's quote against the source it names. When any is not there, the
    edit is refused with every problem, each pointing at the field to fix."""
    sources = (await _sources(session, [question_id])).get(question_id, [])
    texts = {chunk.id: chunk.text for _, chunk in sources}
    checks = [
        check_quote(point.evidence_quote, point.chunk_id, texts.get(point.chunk_id))
        for point in points
    ]
    problems = []
    for index, (point, check) in enumerate(zip(points, checks, strict=True)):
        if check.grounded:
            continue
        # Shaped like FastAPI's own validation errors, so a form can show each by its field
        field = "evidence_quote" if point.chunk_id in texts else "chunk_id"
        problems.append(
            {
                "type": "quote_not_in_source",
                "loc": ["body", "key_points", index, field],
                "msg": check.problem,
                "input": getattr(point, field),
            }
        )
    if problems:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, problems)
    return checks


async def _embed(embedder: Embedder | None, text: str) -> list[float] | None:
    """A question text's embedding for the duplicate check, or None while the model is away."""
    if embedder is None:
        return None
    try:
        [vector] = await embedder.embed_documents([text])
    except EmbeddingError as exc:
        log.warning("the question's new text goes without an embedding: %s", exc)
        return None
    return vector


async def _detail(session: AsyncSession, question_id: int) -> QuestionDetailOut:
    row = (
        await session.execute(
            select(Question, Topic.name, source_updated().label("updated"))
            .outerjoin(Topic, Topic.id == Question.topic_id)
            .where(Question.id == question_id)
            .execution_options(populate_existing=True)
        )
    ).first()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "question not found")
    question, topic, updated = row
    sources = (await _sources(session, [question_id])).get(question_id, [])
    rating = (await question_ratings(session, [question_id])).get(question_id)
    return QuestionDetailOut(
        **_question_out(question, topic, updated, sources, rating).model_dump(),
        reference_answer=question.reference_answer,
        key_points=[KeyPointOut.model_validate(point) for point in question.key_points],
        misconceptions=question.misconceptions,
        validation=question.validation,
        generator_model=question.generator_model,
        prompt_version=question.prompt_version,
        usage=question.usage,
        sources=[
            SourceOut(
                chunk_id=chunk.id,
                document_id=document.id,
                document_title=document.title,
                citation=citation(document, chunk),
                link=source_link(document, chunk),
                section=chunk.section,
                superseded=chunk.superseded_at is not None,
                text=chunk.text,
            )
            for document, chunk in sources
        ],
    )


@router.get("/topics")
async def list_topics(
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[TopicOut]:
    """The topic map, the topics with the most passages behind them first."""
    chunks = (
        select(ChunkTopic.topic_id, func.count().label("chunks"))
        .join(Chunk, Chunk.id == ChunkTopic.chunk_id)
        .where(Chunk.superseded_at.is_(None))
        .group_by(ChunkTopic.topic_id)
        .subquery()
    )
    questions = (
        select(
            Question.topic_id,
            func.count().label("questions"),
            func.count().filter(Question.status == "accepted").label("accepted"),
        )
        .where(Question.topic_id.is_not(None))
        .group_by(Question.topic_id)
        .subquery()
    )
    chunk_count = func.coalesce(chunks.c.chunks, 0)
    rows = await session.execute(
        select(
            Topic,
            chunk_count,
            func.coalesce(questions.c.questions, 0),
            func.coalesce(questions.c.accepted, 0),
        )
        .outerjoin(chunks, chunks.c.topic_id == Topic.id)
        .outerjoin(questions, questions.c.topic_id == Topic.id)
        .order_by(chunk_count.desc(), Topic.name)
        .limit(limit)
        .offset(offset)
    )
    return [
        TopicOut(
            id=topic.id,
            name=topic.name,
            tags=topic.tags,
            chunk_count=chunk_total,
            question_count=question_total,
            accepted_count=accepted,
        )
        for topic, chunk_total, question_total, accepted in rows.tuples()
    ]


@router.post(
    "/topics/build",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(local_map_only)],
)
async def build_topic_map(session: SessionDep, response: Response) -> TopicMapOut:
    """Queue a build of the topic map: the chunks without tags are tagged, then every tag is
    grouped into topics. Returns 202 with the job the worker will run.

    One build at a time: while a job is still queued or running it is returned unchanged,
    since it will tag whatever is untagged when it gets there.
    """
    chunks, tagged = (
        await session.execute(
            select(func.count(), func.count(ChunkTags.chunk_id))
            .select_from(Chunk)
            .outerjoin(ChunkTags, ChunkTags.chunk_id == Chunk.id)
            .where(Chunk.superseded_at.is_(None))
        )
    ).one()
    active = await session.scalar(
        select(Job)
        .where(Job.kind == "topics", Job.status.in_(queue.ACTIVE_STATUSES))
        .order_by(Job.id.desc())
        .limit(1)
    )
    if active is not None:
        return TopicMapOut(
            message=f"already {active.status}",
            untagged=chunks - tagged,
            job=JobOut.model_validate(active),
        )
    if not chunks:
        # Nothing to tag or group, so nothing is written down
        response.status_code = status.HTTP_200_OK
        return TopicMapOut(
            message="nothing to build it from; add material first", untagged=0, job=None
        )

    job = Job(kind="topics")
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return TopicMapOut(message="queued", untagged=chunks - tagged, job=JobOut.model_validate(job))

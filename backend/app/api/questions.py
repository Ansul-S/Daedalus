"""Question endpoints: start a batch, read what came out of it, and browse the topic map.

Starting a batch only writes down the plan; `make worker` (or `make generate`) does the
writing. Checking a question and testing it for duplicates both run on the local models, so
starting a batch is a local-only endpoint, like adding material.

A question is shown with everything that stands behind it: the chunks it was written from,
cited the same way search results are, the key points an answer has to cover with the quote
that proves each one, and the report from every check it went through, whether it passed or
failed.
"""

from datetime import datetime
from typing import Annotated, Any, Literal, get_args

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import ColumnElement, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.documents import JobOut
from app.api.search import citation, source_link
from app.core.config import Settings, get_settings
from app.db.models import (
    QUESTION_STATUSES,
    Chunk,
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
from app.questions.batch import start_run
from app.questions.generation import Style

router = APIRouter(tags=["questions"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]

Status = Literal["accepted", "rejected", "retired"]
assert set(get_args(Status)) == set(QUESTION_STATUSES)


def local_only(settings: SettingsDep) -> None:
    if settings.environment != "local":
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Writing questions runs locally; start a batch from your own machine",
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


def _filters(
    status: Status | None,
    topic_id: int | None,
    style: Style | None,
    difficulty: int | None,
    document_id: int | None,
    updated: bool | None,
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
    question: Question, topic: str | None, updated: bool, sources: list[tuple[Document, Chunk]]
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
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> QuestionsOut:
    """The questions in the library, newest first. Every filter is optional."""
    conditions = _filters(status, topic_id, style, difficulty, document_id, updated)
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
    sources = await _sources(session, [question.id for question, _, _ in found])
    return QuestionsOut(
        total=total or 0,
        limit=limit,
        offset=offset,
        results=[
            _question_out(question, topic, updated, sources.get(question.id, []))
            for question, topic, updated in found
        ],
    )


@router.get("/questions/{question_id}")
async def get_question(question_id: int, session: SessionDep) -> QuestionDetailOut:
    """One question with its sources, key points and validation report."""
    row = (
        await session.execute(
            select(Question, Topic.name, source_updated().label("updated"))
            .outerjoin(Topic, Topic.id == Question.topic_id)
            .where(Question.id == question_id)
        )
    ).first()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "question not found")
    question, topic, updated = row
    sources = (await _sources(session, [question_id])).get(question_id, [])
    return QuestionDetailOut(
        **_question_out(question, topic, updated, sources).model_dump(),
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

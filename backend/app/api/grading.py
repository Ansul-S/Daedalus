"""Answering a question and getting it graded.

An answer is written down before it is graded, so it is kept even when no model can grade
it; such an attempt carries a failed grade saying why, and can be graded again later. A
grade is shown with each key point's text next to its label, and each claim with the passage
behind its verdict, cited and linked the way search results are.

A graded answer also reschedules its question: the score earns a rating, and the rating
decides the practice day the question comes back (`app.scheduling`). Only an attempt's first
successful grade does this, so grading an answer again never counts it twice. What that grade
earned (XP, the level it reached and any coins) comes with the attempt from then on.

Grading runs on cloud models first, so unlike writing questions it also works in production.
The grading model is built once per process: its pacer has to remember the minute's requests
across answers. It starts from what the day has already spent, and the day's allowance
refills while the process runs, however long that is.
"""

from datetime import date, datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, field_validator
from pydantic_ai.models import Model
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.practice import EarnedOut, earned_out
from app.api.search import citation, source_link
from app.core.config import Settings, get_settings
from app.db.models import Attempt, Chunk, Document, Grade, Question, Review
from app.db.session import get_session
from app.grading.grader import grade_attempt, spent_today
from app.llm.models import grading_model
from app.scheduling.progress import load, walk
from app.scheduling.schedule import rating_name, record_review

router = APIRouter(tags=["grading"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]

# About two thousand tokens: room for a thorough answer, not for pasting in a paper
MAX_ANSWER_CHARS = 8000
# A day: a stopwatch left running overnight, not an answer
MAX_SECONDS = 24 * 60 * 60
# An hour: interview mode gives three minutes
MAX_TIME_LIMIT = 60 * 60


async def get_grader(request: Request, session: SessionDep, settings: SettingsDep) -> Model:
    grader = getattr(request.app.state, "grader", None)
    if grader is None:
        grader = grading_model(settings, await spent_today(session))
        request.app.state.grader = grader
    return grader


GraderDep = Annotated[Model, Depends(get_grader)]


class AnswerIn(BaseModel):
    answer: str = Field(max_length=MAX_ANSWER_CHARS)
    # How long the answer took, in seconds
    seconds: float | None = Field(None, ge=0, le=MAX_SECONDS)
    # The limit it was answered under in interview mode, in seconds
    time_limit: int | None = Field(None, gt=0, le=MAX_TIME_LIMIT)

    @field_validator("answer")
    @classmethod
    def not_blank(cls, answer: str) -> str:
        if not answer.strip():
            raise ValueError("the answer is empty")
        return answer


class KeyPointGradeOut(BaseModel):
    id: str
    text: str
    weight: int
    status: str
    answer_quote: str
    # Whether the quoted words are really in the answer; null for a missing point
    quote_found: bool | None


class ClaimOut(BaseModel):
    claim: str
    verdict: str
    why: str
    chunk_id: int | None
    citation: str | None
    link: str | None


class GradeOut(BaseModel):
    id: int
    status: str
    error: str | None
    grader_model: str | None
    prompt_version: str
    score: float | None
    coverage: float | None
    contradicted: int | None
    clarity: int | None
    key_points: list[KeyPointGradeOut]
    claims: list[ClaimOut]
    strengths: list[str]
    gaps: list[str]
    errors: list[str]
    improved_answer: str | None
    follow_up: str | None
    usage: dict[str, Any]
    seconds: float | None
    created_at: datetime


class ReviewOut(BaseModel):
    rating: Literal["again", "hard", "good", "easy"]
    # The practice day the answer counted for, and the one the question is due again
    day: date
    due: date
    # Days from the one to the other
    interval: int


class AttemptOut(BaseModel):
    id: int
    question_id: int
    question: str
    answer: str
    seconds: float | None
    time_limit: int | None
    created_at: datetime
    # Oldest first; the last one is the latest
    grades: list[GradeOut]
    # What the answer did to the schedule; null until it is graded
    review: ReviewOut | None
    # What its first successful grade earned: XP, the level it reached and any coins; null
    # until it is graded
    earned: EarnedOut | None


async def _cited(session: AsyncSession, grades: list[Grade]) -> dict[int, tuple[Document, Chunk]]:
    """Every chunk the grades' claims cite, in one query."""
    chunk_ids = {
        claim["chunk_id"]
        for grade in grades
        for claim in grade.claims
        if claim.get("chunk_id") is not None
    }
    if not chunk_ids:
        return {}
    rows = await session.execute(
        select(Document, Chunk)
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.id.in_(chunk_ids))
    )
    return {chunk.id: (document, chunk) for document, chunk in rows.tuples()}


def _grade_out(
    grade: Grade, question: Question, cited: dict[int, tuple[Document, Chunk]]
) -> GradeOut:
    key_points = [
        KeyPointGradeOut(
            id=label["id"],
            text=point["text"],
            weight=point["weight"],
            status=label["status"],
            answer_quote=label["answer_quote"],
            quote_found=label.get("quote_found"),
        )
        for label, point in zip(grade.key_points, question.key_points, strict=False)
    ]
    claims = []
    for claim in grade.claims:
        source = cited.get(claim["chunk_id"]) if claim.get("chunk_id") is not None else None
        claims.append(
            ClaimOut(
                claim=claim["claim"],
                verdict=claim["verdict"],
                why=claim["why"],
                chunk_id=claim.get("chunk_id"),
                citation=citation(*source) if source else None,
                link=source_link(*source) if source else None,
            )
        )
    return GradeOut(
        id=grade.id,
        status=grade.status,
        error=grade.error,
        grader_model=grade.grader_model,
        prompt_version=grade.prompt_version,
        score=grade.score,
        coverage=grade.coverage,
        contradicted=grade.contradicted,
        clarity=grade.clarity,
        key_points=key_points,
        claims=claims,
        strengths=grade.strengths,
        gaps=grade.gaps,
        errors=grade.errors,
        improved_answer=grade.improved_answer,
        follow_up=grade.follow_up,
        usage=grade.usage,
        seconds=grade.seconds,
        created_at=grade.created_at,
    )


def _review_out(review: Review | None) -> ReviewOut | None:
    if review is None:
        return None
    return ReviewOut(
        rating=rating_name(review.rating),
        day=review.day,
        due=review.due,
        interval=(review.due - review.day).days,
    )


async def _attempts_out(session: AsyncSession, attempts: list[Attempt]) -> list[AttemptOut]:
    cited = await _cited(session, [grade for attempt in attempts for grade in attempt.grades])
    # What an answer earned depends on every answer before it: the history is walked through.
    steps = {}
    if any(attempt.review for attempt in attempts):
        steps = {step.answer.review_id: step for step in walk(*await load(session)).steps}
    return [
        AttemptOut(
            id=attempt.id,
            question_id=attempt.question_id,
            question=attempt.question.text,
            answer=attempt.answer,
            seconds=attempt.seconds,
            time_limit=attempt.time_limit,
            created_at=attempt.created_at,
            grades=[_grade_out(grade, attempt.question, cited) for grade in attempt.grades],
            review=_review_out(attempt.review),
            earned=earned_out(steps[attempt.review.id]) if attempt.review else None,
        )
        for attempt in attempts
    ]


# Everything an attempt is shown with
ATTEMPT_PARTS = (
    selectinload(Attempt.grades),
    selectinload(Attempt.question),
    selectinload(Attempt.review),
)


async def _attempt(session: AsyncSession, attempt_id: int) -> Attempt:
    attempt = await session.scalar(
        select(Attempt)
        .options(*ATTEMPT_PARTS)
        .where(Attempt.id == attempt_id)
        .execution_options(populate_existing=True)
    )
    if attempt is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "attempt not found")
    return attempt


@router.post("/questions/{question_id}/attempts", status_code=status.HTTP_201_CREATED)
async def answer_question(
    question_id: int,
    body: AnswerIn,
    session: SessionDep,
    grader: GraderDep,
    settings: SettingsDep,
) -> AttemptOut:
    """Answer a question, have the answer graded, and reschedule the question. Only questions
    in the library can be answered: one that was rejected or retired is not there to practise."""
    question = await session.get(Question, question_id)
    if question is None or question.status != "accepted":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "question not found")
    attempt = Attempt(
        question_id=question_id,
        answer=body.answer,
        seconds=body.seconds,
        time_limit=body.time_limit,
    )
    session.add(attempt)
    # The answer is kept whatever becomes of its grading.
    await session.commit()
    grade = await grade_attempt(session, grader, attempt)
    await record_review(session, grade, settings.practice_zone)
    await session.commit()
    return (await _attempts_out(session, [await _attempt(session, attempt.id)]))[0]


@router.post("/attempts/{attempt_id}/grades", status_code=status.HTTP_201_CREATED)
async def grade_again(
    attempt_id: int, session: SessionDep, grader: GraderDep, settings: SettingsDep
) -> AttemptOut:
    """Grade an attempt again, after a failed grade or with a changed grader. Every earlier
    grade is kept. The first successful grade reschedules the question; later ones don't."""
    attempt = await _attempt(session, attempt_id)
    grade = await grade_attempt(session, grader, attempt)
    await record_review(session, grade, settings.practice_zone)
    await session.commit()
    return (await _attempts_out(session, [await _attempt(session, attempt_id)]))[0]


@router.get("/attempts/{attempt_id}")
async def get_attempt(attempt_id: int, session: SessionDep) -> AttemptOut:
    """One attempt with every grade it was given."""
    return (await _attempts_out(session, [await _attempt(session, attempt_id)]))[0]


@router.get("/questions/{question_id}/attempts")
async def list_attempts(
    question_id: int,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[AttemptOut]:
    """The answers given to a question, newest first."""
    if await session.get(Question, question_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "question not found")
    attempts = await session.scalars(
        select(Attempt)
        .options(*ATTEMPT_PARTS)
        .where(Attempt.question_id == question_id)
        .order_by(Attempt.created_at.desc(), Attempt.id.desc())
        .limit(limit)
        .offset(offset)
    )
    return await _attempts_out(session, list(attempts))

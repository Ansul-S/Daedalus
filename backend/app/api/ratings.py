"""Ratings: your own judgement of a question or of a grade.

A question is rated good or poor and a grade fair or unfair, each with an optional note on
why. The ratings are evaluation data: they show which questions the generator got wrong, and
where the grader goes wrong in real use, now that the calibration answers can no longer be
used to tune it. Every rating is kept. The latest rating of a question or grade is the one
that stands, and it comes with the question or grade wherever that is shown.
"""

from collections.abc import Iterable
from datetime import datetime
from typing import Annotated, Literal, Self

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import ScalarSelect, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import InstrumentedAttribute

from app.db.models import Grade, Question, Rating
from app.db.session import get_session

router = APIRouter(tags=["ratings"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]

# A sentence or two
MAX_NOTE_CHARS = 500

# What the question list can be narrowed to: the questions whose latest rating is good, or
# poor, or the ones never rated
RatingFilter = Literal["good", "poor", "unrated"]
RATING_VALUES = {"good": 1, "poor": -1}


class RatingIn(BaseModel):
    """A rating of one question or of one grade."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    question_id: int | None = Field(None, description="The question rated; or give grade_id")
    grade_id: int | None = Field(None, description="The grade rated; or give question_id")
    value: Literal[1, -1] = Field(
        description="1 for a good question or a fair grade, -1 for a poor question or an "
        "unfair grade"
    )
    note: str | None = Field(None, max_length=MAX_NOTE_CHARS, description="Why, in a sentence")

    @model_validator(mode="after")
    def rates_one_thing(self) -> Self:
        if (self.question_id is None) == (self.grade_id is None):
            raise ValueError("rate one thing: give either question_id or grade_id")
        return self


class RatingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    # What was rated: a question or a grade
    question_id: int | None
    grade_id: int | None
    # 1 for a good question or a fair grade, -1 for a poor question or an unfair grade
    value: Literal[1, -1]
    note: str | None
    created_at: datetime


@router.post("/ratings", status_code=status.HTTP_201_CREATED)
async def add_rating(body: RatingIn, session: SessionDep) -> RatingOut:
    """Rate a question good or poor, or a grade fair or unfair, with an optional note on why.

    A rating never replaces an earlier one; the latest is the one that stands. Any question
    can be rated, whatever its status: a rejected question rated good is a check that turned
    down too much. A grade can be rated once it has graded the answer: a failed grade has no
    verdict to judge.
    """
    if body.question_id is not None and await session.get(Question, body.question_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "question not found")
    if body.grade_id is not None:
        grade = await session.get(Grade, body.grade_id)
        if grade is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "grade not found")
        if grade.status != "graded":
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                "a failed grade has no verdict to rate; grade the answer again first",
            )
    rating = Rating(
        question_id=body.question_id,
        grade_id=body.grade_id,
        value=body.value,
        # A blank note says nothing
        note=body.note or None,
    )
    session.add(rating)
    await session.commit()
    await session.refresh(rating)
    return RatingOut.model_validate(rating)


def latest_rating() -> ScalarSelect[int]:
    """A question's latest rating, 1 or -1, or null for a question never rated: a condition
    for the question list, derived like `source_updated`."""
    return (
        select(Rating.value)
        .where(Rating.question_id == Question.id)
        .order_by(Rating.id.desc())
        .limit(1)
        .scalar_subquery()
    )


async def question_ratings(session: AsyncSession, ids: Iterable[int]) -> dict[int, RatingOut]:
    """The latest rating of each of these questions that has one, in one query."""
    return await _latest(session, Rating.question_id, ids)


async def grade_ratings(session: AsyncSession, ids: Iterable[int]) -> dict[int, RatingOut]:
    """The latest rating of each of these grades that has one, in one query."""
    return await _latest(session, Rating.grade_id, ids)


async def _latest(
    session: AsyncSession, rated: InstrumentedAttribute[int | None], ids: Iterable[int]
) -> dict[int, RatingOut]:
    wanted = list(ids)
    if not wanted:
        return {}
    ratings = await session.scalars(
        select(Rating).where(rated.in_(wanted)).order_by(rated, Rating.id.desc()).distinct(rated)
    )
    return {getattr(rating, rated.key): RatingOut.model_validate(rating) for rating in ratings}

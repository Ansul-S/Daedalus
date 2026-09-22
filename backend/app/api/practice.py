"""Practice: which question to answer next.

The question comes the way an interviewer would ask it, without its reference answer or key
points, and with the reason it was picked: due for review, new from the weakest topic, or
practice ahead of the schedule. Answering it goes through `POST /questions/{id}/attempts`,
which grades the answer and reschedules the question.
"""

from datetime import UTC, date, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.questions import QuestionOut, _question_out, _sources
from app.core.config import Settings, get_settings
from app.db.models import Question, Topic, source_updated
from app.db.session import get_session
from app.scheduling.picker import Pick, Reason, next_question
from app.scheduling.schedule import practice_day

router = APIRouter(tags=["practice"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


class PracticeOut(BaseModel):
    question: QuestionOut
    reason: Reason
    # Why this question now, in a phrase for the practice page
    why: str
    # The practice day it was due; null for a question never practised
    due: date | None
    # How likely it is to be answered well today, by the memory model: 0 to 1, and 0 for a
    # question never practised
    recall: float
    # Mastery of its topic, 0 to 1
    topic_mastery: float
    # Questions due today or earlier, this one included, and questions never practised
    due_count: int
    new_count: int


def why(pick: Pick, topic: str | None, today: date) -> str:
    if pick.reason == "due" and pick.due is not None:
        late = (today - pick.due).days
        if late == 0:
            return "due for review today"
        return f"due for review, {plural(late, 'day')} late"
    if pick.reason == "new":
        return f"new, from your weakest topic: {topic}" if topic else "new"
    return "practising ahead: the question you are likeliest to have forgotten"


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


@router.get("/practice/next")
async def practice_next(session: SessionDep, settings: SettingsDep) -> PracticeOut:
    """The question to practise next, without its answer, and why it was picked."""
    today = practice_day(datetime.now(UTC), settings.practice_zone)
    pick = await next_question(session, today)
    if pick is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "no questions to practise yet; generate some first"
        )
    question, topic, updated = (
        await session.execute(
            select(Question, Topic.name, source_updated().label("updated"))
            .outerjoin(Topic, Topic.id == Question.topic_id)
            .where(Question.id == pick.question_id)
        )
    ).one()
    sources = (await _sources(session, [question.id])).get(question.id, [])
    return PracticeOut(
        question=_question_out(question, topic, updated, sources),
        reason=pick.reason,
        why=why(pick, topic, today),
        due=pick.due,
        recall=round(pick.retrievability, 4),
        topic_mastery=round(pick.topic_mastery, 4),
        due_count=pick.due_count,
        new_count=pick.new_count,
    )

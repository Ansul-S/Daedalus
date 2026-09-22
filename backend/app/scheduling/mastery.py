"""How well each question and each topic is known on a given practice day.

A question's mastery is its latest score times the chance of recalling it that day, so it
fades while the question goes unpractised, the way the memory model says recall fades. A
question never practised has none. A topic's mastery is the mean over its questions in the
library: roughly what they would score, on average, if all of them were asked that day.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Card, Question, Review
from app.scheduling.schedule import retrievability


@dataclass(frozen=True)
class Standing:
    """Where one question in the library stands on a practice day."""

    question_id: int
    topic_id: int | None
    difficulty: int
    # The latest review's score, the memory state and the day it is due; None when the
    # question has never been practised
    score: float | None
    state: dict[str, Any] | None
    due: date | None
    # The chance of recalling it that day, 0 to 1; 0 when never practised
    retrievability: float

    @property
    def practised(self) -> bool:
        return self.state is not None

    @property
    def mastery(self) -> float:
        return (self.score or 0.0) * self.retrievability


async def standings(session: AsyncSession, day: date) -> list[Standing]:
    """Every question in the library, practised or not, as it stands on `day`."""
    latest = (
        select(Review.question_id, Review.score)
        .distinct(Review.question_id)
        .order_by(Review.question_id, Review.id.desc())
        .subquery()
    )
    rows = await session.execute(
        select(
            Question.id,
            Question.topic_id,
            Question.difficulty,
            latest.c.score,
            Card.state,
            Card.due,
        )
        .outerjoin(Card, Card.question_id == Question.id)
        .outerjoin(latest, latest.c.question_id == Question.id)
        .where(Question.status == "accepted")
        .order_by(Question.id)
    )
    return [
        Standing(
            question_id=question_id,
            topic_id=topic_id,
            difficulty=difficulty,
            score=score,
            state=state,
            due=due,
            retrievability=retrievability(state, day) if state is not None else 0.0,
        )
        for question_id, topic_id, difficulty, score, state, due in rows.tuples()
    ]


def topic_mastery(questions: Iterable[Standing]) -> dict[int | None, float]:
    """Each topic's mastery, the mean over its questions. Questions without a topic are kept
    together under None."""
    grouped: dict[int | None, list[float]] = {}
    for question in questions:
        grouped.setdefault(question.topic_id, []).append(question.mastery)
    return {topic: sum(values) / len(values) for topic, values in grouped.items()}

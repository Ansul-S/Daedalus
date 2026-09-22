"""Which question to practise next.

1. **Due:** the question most overdue for review, when any is due that day or earlier.
2. **New:** otherwise a question never practised, from the weakest topic: the lowest mastery,
   and among equals the topic practised least, so a newcomer meets every topic in turn. Within
   the topic, the easiest question first.
3. **Ahead:** otherwise, practising ahead of the schedule, the question most likely to have
   been forgotten.

Only questions in the library are picked, so a retired question drops out.
"""

from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

from app.scheduling.mastery import Standing, standings, topic_mastery

Reason = Literal["due", "new", "ahead"]


@dataclass(frozen=True)
class Pick:
    question_id: int
    reason: Reason
    # The practice day it was due; None for a question never practised
    due: date | None
    # The chance of recalling it that day, and the mastery of its topic, both 0 to 1
    retrievability: float
    topic_mastery: float
    # Questions due that day or earlier, this one included, and questions never practised
    due_count: int
    new_count: int


async def next_question(session: AsyncSession, day: date) -> Pick | None:
    """The question to practise next on `day`, or None when the library is empty."""
    return choose(await standings(session, day), day)


def choose(questions: list[Standing], day: date) -> Pick | None:
    if not questions:
        return None
    mastery = topic_mastery(questions)
    due = [question for question in questions if question.due is not None and question.due <= day]
    new = [question for question in questions if not question.practised]

    def pick(question: Standing, reason: Reason) -> Pick:
        return Pick(
            question_id=question.question_id,
            reason=reason,
            due=question.due,
            retrievability=question.retrievability,
            topic_mastery=mastery[question.topic_id],
            due_count=len(due),
            new_count=len(new),
        )

    if due:
        return pick(min(due, key=lambda q: (q.due, q.retrievability, q.question_id)), "due")
    if new:
        practised = Counter(question.topic_id for question in questions if question.practised)
        weakest = min(
            {question.topic_id for question in new},
            key=lambda topic: (mastery[topic], practised[topic], topic is None, topic or 0),
        )
        return pick(
            min(
                (question for question in new if question.topic_id == weakest),
                key=lambda q: (q.difficulty, q.question_id),
            ),
            "new",
        )
    return pick(min(questions, key=lambda q: (q.retrievability, q.due, q.question_id)), "ahead")

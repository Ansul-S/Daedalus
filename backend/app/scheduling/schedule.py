"""When a practised question comes back.

A graded answer earns an FSRS rating from its score, and py-fsrs reschedules the question
with it. The thresholds were set on the 75 hand-graded calibration answers, and the
scheduler's settings on simulated review histories:

- Again below 0.4, Hard below 0.7, Good below 0.9, Easy from 0.9. The grader marks a partial
  answer about one step lower than a person does, so 0.4 is about half the key points as a
  person would judge them; every calibration answer graded 2 out of 10 or less by hand falls
  below it.
- Any claim the sources contradict makes the rating Again, whatever the score. Once a question
  is known, Hard, Good and Easy all send it weeks away; only Again brings a confident mistake
  back within days.
- No learning steps. py-fsrs's defaults ask a new question again after 1 and then 10 minutes,
  the pace of flashcards, and would keep asking a partly answered one every few minutes.
  Without steps the shortest wait is a day.
- At most 30 days between reviews, so everything practised comes back within a month.
  Unbounded, four good answers in a row put the next review months away.
- Time is counted in practice days. A day starts at 04:00 in `PRACTICE_TIMEZONE`, and FSRS
  sees each day as its midnight in UTC, so the days between two answers are calendar days
  whatever the hour. py-fsrs itself counts whole 24-hour spans: an answer given 20 hours after
  the last one would count as the same day, which barely moves the question on.
"""

from datetime import UTC, date, datetime, time, timedelta, tzinfo
from typing import Any

import fsrs
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Attempt, Card, Grade, Review

AGAIN_BELOW = 0.4
GOOD_FROM = 0.7
EASY_FROM = 0.9
DESIRED_RETENTION = 0.9
MAXIMUM_INTERVAL = 30
# When a practice day starts, in local time
DAY_STARTS = timedelta(hours=4)


def scheduler(fuzz: bool = True) -> fsrs.Scheduler:
    """A scheduler with the settings above. Fuzzing moves an interval of three days or more by
    a day or two, so that questions answered together don't all come back together."""
    return fsrs.Scheduler(
        desired_retention=DESIRED_RETENTION,
        learning_steps=(),
        relearning_steps=(),
        maximum_interval=MAXIMUM_INTERVAL,
        enable_fuzzing=fuzz,
    )


SCHEDULER = scheduler()


def rating_for(score: float, contradicted: int) -> fsrs.Rating:
    """The rating a graded answer earns."""
    if contradicted > 0 or score < AGAIN_BELOW:
        return fsrs.Rating.Again
    if score < GOOD_FROM:
        return fsrs.Rating.Hard
    if score < EASY_FROM:
        return fsrs.Rating.Good
    return fsrs.Rating.Easy


def rating_name(rating: int) -> str:
    return fsrs.Rating(rating).name.lower()


def practice_day(when: datetime, zone: tzinfo) -> date:
    """The practice day a moment belongs to. A day starts at 04:00 local time, so a session
    that runs past midnight counts as one day."""
    return (when.astimezone(zone) - DAY_STARTS).date()


def moment(day: date) -> datetime:
    """A practice day as FSRS sees it: its midnight in UTC, so that consecutive days are always
    exactly 24 hours apart."""
    return datetime.combine(day, time(), tzinfo=UTC)


def review(
    card: fsrs.Card, rating: fsrs.Rating, day: date, scheduler: fsrs.Scheduler | None = None
) -> fsrs.Card:
    """The card after an answer given on `day`. Its `due` is the midnight of a practice day."""
    reviewed, _ = (scheduler or SCHEDULER).review_card(card, rating, review_datetime=moment(day))
    return reviewed


def retrievability(state: dict[str, Any], day: date) -> float:
    """How likely a practised question is to be answered well on `day`, 0 to 1, by the memory
    model."""
    return SCHEDULER.get_card_retrievability(fsrs.Card.from_dict(state), moment(day))


async def record_review(session: AsyncSession, grade: Grade, zone: tzinfo) -> Review | None:
    """Reschedule the question a grade is for. Only a successful grade reschedules, and only
    the first one an attempt gets: grading an answer again doesn't make it count twice.
    Returns the review, or None when there is none to make."""
    if grade.status != "graded" or grade.score is None or grade.contradicted is None:
        return None
    if await session.scalar(select(Review.id).where(Review.attempt_id == grade.attempt_id)):
        return None
    question_id, answered_at = (
        await session.execute(
            select(Attempt.question_id, Attempt.created_at).where(Attempt.id == grade.attempt_id)
        )
    ).one()
    stored = await session.scalar(
        select(Card).where(Card.question_id == question_id).with_for_update()
    )
    card = fsrs.Card.from_dict(stored.state) if stored else fsrs.Card(card_id=question_id)
    # The answer counts for the day it was given. One graded late, after a newer answer was
    # already reviewed, counts for that answer's day, so the schedule never runs backwards.
    day = practice_day(answered_at, zone)
    if card.last_review is not None:
        day = max(day, card.last_review.date())
    rating = rating_for(grade.score, grade.contradicted)
    card = review(card, rating, day)
    due = card.due.date()
    if stored is None:
        session.add(Card(question_id=question_id, state=card.to_dict(), due=due))
    else:
        stored.state, stored.due = card.to_dict(), due
    reviewed = Review(
        question_id=question_id,
        attempt_id=grade.attempt_id,
        grade_id=grade.id,
        rating=int(rating),
        score=grade.score,
        day=day,
        due=due,
    )
    session.add(reviewed)
    await session.flush()
    return reviewed

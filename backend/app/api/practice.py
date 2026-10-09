"""Practice: which question to answer next, and what practice has earned.

The question comes the way an interviewer would ask it, without its reference answer or key
points, and with the reason it was picked: due for review, new from the weakest topic, or
practice ahead of the schedule. Answering it goes through `POST /questions/{id}/attempts`,
which grades the answer and reschedules the question.

What practice has earned (XP, a level, the streak and coins), the labyrinth on the dashboard
and its charts are all worked out from the review history when they are asked for. So is what
the daily limits on grading leave (`app.grading.limits`).

All of it is the user's own (`app.api.users`): the schedule, the history and what it earned
are theirs, and nobody else's answers count. A user can delete all of it.
"""

from datetime import UTC, date, datetime, timedelta
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.questions import QuestionOut, _question_out, _sources
from app.api.ratings import question_ratings
from app.api.users import UserDep
from app.core.config import Settings, get_settings
from app.db.models import (
    Attempt,
    Card,
    Grade,
    Interview,
    Question,
    Rating,
    Review,
    Topic,
    source_updated,
)
from app.db.session import get_session
from app.grading.limits import Limit, allowance
from app.interview import places
from app.scheduling import labyrinth
from app.scheduling.mastery import standings
from app.scheduling.picker import Pick, Reason, next_question
from app.scheduling.progress import (
    LEVELS,
    CoinState,
    Level,
    Progress,
    Step,
    load,
    progress,
    walk,
)
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
async def practice_next(
    session: SessionDep, settings: SettingsDep, user_id: UserDep
) -> PracticeOut:
    """The question to practise next, without its answer, and why it was picked."""
    today = practice_day(datetime.now(UTC), settings.practice_zone)
    pick = await next_question(session, user_id, today)
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
    rating = (await question_ratings(session, user_id, [question.id])).get(question.id)
    return PracticeOut(
        question=_question_out(question, topic, updated, sources, rating),
        reason=pick.reason,
        why=why(pick, topic, today),
        due=pick.due,
        recall=round(pick.retrievability, 4),
        topic_mastery=round(pick.topic_mastery, 4),
        due_count=pick.due_count,
        new_count=pick.new_count,
    )


class LevelOut(BaseModel):
    number: int
    name: str
    # How many levels there are
    of: int
    # The XP it starts at, and where the next level starts and what it is called; null at
    # the top
    start: int
    next_start: int | None
    next_name: str | None


class CoinOut(BaseModel):
    id: str
    name: str
    glyph: str
    # What mints it
    condition: str
    # The practice day it was minted; null while it is still to earn
    minted_on: date | None
    # How far along a coin still to earn is, where that can be counted
    have: int | None
    need: int | None


class StreakOut(BaseModel):
    # Practice days in a row, up to today, or up to yesterday while today has no answer yet
    days: int
    # Whether today has a graded answer yet
    today: bool
    best: int


class ProgressOut(BaseModel):
    xp: int
    level: LevelOut
    streak: StreakOut
    # Graded answers
    answers: int
    # Every coin, in the order the shelf shows them
    coins: list[CoinOut]


class XpOut(BaseModel):
    """The XP an answer earned, part by part: ten times its score, five for answering, half
    as much again for a question that was due, five inside an interview limit, and, on the
    day's first answer, the streak's length that day, up to ten."""

    score: int
    answered: int
    due: int
    interview: int
    streak: int


class EarnedOut(BaseModel):
    # The XP an answer earned, and its parts
    xp: int
    parts: XpOut
    # XP after it, the level that reached, and whether that level is new
    total_xp: int
    level: LevelOut
    level_up: bool
    # The streak on the day it counted for
    streak: int
    # The coins it minted
    coins: list[CoinOut]


def _level_out(level: Level) -> LevelOut:
    return LevelOut(
        number=level.number,
        name=level.name,
        of=len(LEVELS),
        start=level.start,
        next_start=level.next_start,
        next_name=level.next_name,
    )


def _coin_out(state: CoinState) -> CoinOut:
    return CoinOut(
        id=state.coin.id,
        name=state.coin.name,
        glyph=state.coin.glyph,
        condition=state.coin.condition,
        minted_on=state.minted_on,
        have=state.have,
        need=state.need,
    )


def earned_out(step: Step) -> EarnedOut:
    return EarnedOut(
        xp=step.xp.total,
        parts=XpOut(
            score=step.xp.score,
            answered=step.xp.answered,
            due=step.xp.due,
            interview=step.xp.interview,
            streak=step.xp.streak,
        ),
        total_xp=step.total,
        level=_level_out(step.level),
        level_up=step.level_up,
        streak=step.streak,
        coins=[_coin_out(CoinState(coin, step.answer.day)) for coin in step.minted],
    )


def _progress_out(found: Progress) -> ProgressOut:
    return ProgressOut(
        xp=found.xp,
        level=_level_out(found.level),
        streak=StreakOut(days=found.streak.days, today=found.streak.today, best=found.streak.best),
        answers=found.answers,
        coins=[_coin_out(state) for state in found.coins],
    )


class LimitOut(BaseModel):
    # Whose grades it counts: yours in this practice day, or everyone's over the last 24 hours
    scope: Literal["per_user", "in_all"]
    allowed: int
    used: int
    left: int
    # When a grade is allowed again while none is left; null while one is, and when grading
    # is switched off (a limit of 0)
    again_at: datetime | None


class AllowanceOut(BaseModel):
    # Each limit, or null where there is none
    per_user: LimitOut | None
    in_all: LimitOut | None
    # Grades that can still be asked for now: the fewer the limits leave; null without limits
    left: int | None
    # The limit a grade would be refused by now; null while one is allowed
    refused_by: LimitOut | None


def limit_out(limit: Limit) -> LimitOut:
    return LimitOut(
        scope=limit.scope,
        allowed=limit.allowed,
        used=limit.used,
        left=limit.left,
        again_at=limit.again_at,
    )


@router.get("/practice/allowance")
async def practice_allowance(
    session: SessionDep, settings: SettingsDep, user_id: UserDep
) -> AllowanceOut:
    """What the daily limits on grading leave you: your own grades today, and everyone's
    over the last 24 hours."""
    found = await allowance(session, settings, user_id, datetime.now(UTC))
    refused = found.refusal()
    return AllowanceOut(
        per_user=limit_out(found.per_user) if found.per_user else None,
        in_all=limit_out(found.in_all) if found.in_all else None,
        left=min((limit.left for limit in found.limits), default=None),
        refused_by=limit_out(refused) if refused else None,
    )


@router.get("/practice/progress")
async def practice_progress(
    session: SessionDep, settings: SettingsDep, user_id: UserDep
) -> ProgressOut:
    """XP, level, streak and coins, worked out from every graded answer."""
    today = practice_day(datetime.now(UTC), settings.practice_zone)
    return _progress_out(progress(walk(*await load(session, user_id)), today))


class RoomOut(BaseModel):
    topic_id: int
    name: str
    # Where it is: cells are numbered row by row from the top left
    cell: int
    questions: int
    # Questions practised at least once, and questions due today or earlier
    practised: int
    due: int
    # 0 to 1: the mean over its questions of the latest score times the chance of recalling it
    mastery: float


class MapOut(BaseModel):
    columns: int
    rows: int
    rooms: list[RoomOut]
    # Neighbouring cells joined by a passage, the lower-numbered first
    passages: list[tuple[int, int]]
    # The entrance and the Minotaur's room; null when there are no rooms
    entrance: int | None
    lair: int | None
    # The rooms answered in today, in order, and the thread from the entrance through them
    visits: list[int]
    thread: list[int]


@router.get("/practice/map")
async def practice_map(session: SessionDep, settings: SettingsDep, user_id: UserDep) -> MapOut:
    """The labyrinth on the dashboard: a room for each topic with questions in the library,
    how well it is known and what is due in it, the passages between the rooms, today's
    thread through them, and the Minotaur's room, the weakest."""
    today = practice_day(datetime.now(UTC), settings.practice_zone)
    questions = await standings(session, user_id, today)
    rooms = labyrinth.topic_rooms(questions, today)
    maze = labyrinth.carve(len(rooms))
    if maze is None:
        return MapOut(
            columns=labyrinth.COLUMNS,
            rows=0,
            rooms=[],
            passages=[],
            entrance=None,
            lair=None,
            visits=[],
            thread=[],
        )
    topics = await session.execute(
        select(Topic.id, Topic.name).where(Topic.id.in_([room.topic_id for room in rooms]))
    )
    names = {topic_id: name for topic_id, name in topics.tuples()}
    topic_of = {question.question_id: question.topic_id for question in questions}
    answered = await session.scalars(
        select(Review.question_id)
        .where(Review.user_id == user_id, Review.day == today)
        .order_by(Review.id)
    )
    visits = labyrinth.visited((topic_of.get(question_id) for question_id in answered), rooms)
    return MapOut(
        columns=maze.columns,
        rows=maze.rows,
        rooms=[
            RoomOut(
                topic_id=room.topic_id,
                name=names[room.topic_id],
                cell=cell,
                questions=room.questions,
                practised=room.practised,
                due=room.due,
                mastery=round(room.mastery, 4),
            )
            for cell, room in enumerate(rooms)
        ],
        passages=sorted(maze.passages),
        entrance=maze.entrance,
        lair=labyrinth.lair(rooms),
        visits=visits,
        thread=labyrinth.thread(maze, visits),
    )


# What the dashboard's charts show
SCORES_SHOWN = 12
DAYS_SHOWN = 14
DAYS_AHEAD = 7


class ScoreOut(BaseModel):
    attempt_id: int
    question_id: int
    day: date
    score: float


class DayCountOut(BaseModel):
    day: date
    count: int


class StatsOut(BaseModel):
    today: date
    # The latest graded answers' scores, oldest first
    scores: list[ScoreOut]
    # Graded answers on each of the last 14 practice days, today last
    answered: list[DayCountOut]
    # Questions due on each of the next 7 days, today first, counting any overdue
    due: list[DayCountOut]


@router.get("/practice/stats")
async def practice_stats(session: SessionDep, settings: SettingsDep, user_id: UserDep) -> StatsOut:
    """The dashboard's charts: the latest scores, the days practised, and what comes due."""
    today = practice_day(datetime.now(UTC), settings.practice_zone)
    latest = (
        await session.execute(
            select(Review.attempt_id, Review.question_id, Review.day, Review.score)
            .where(Review.user_id == user_id)
            .order_by(Review.id.desc())
            .limit(SCORES_SHOWN)
        )
    ).tuples()
    first = today - timedelta(days=DAYS_SHOWN - 1)
    answered = await session.execute(
        select(Review.day, func.count())
        .where(Review.user_id == user_id, Review.day >= first, Review.day <= today)
        .group_by(Review.day)
    )
    counts = {day: count for day, count in answered.tuples()}
    days = [first + timedelta(days=offset) for offset in range(DAYS_SHOWN)]
    dues = [question.due for question in await standings(session, user_id, today) if question.due]
    ahead = [today + timedelta(days=offset) for offset in range(DAYS_AHEAD)]
    return StatsOut(
        today=today,
        scores=[
            ScoreOut(attempt_id=attempt_id, question_id=question_id, day=day, score=score)
            for attempt_id, question_id, day, score in reversed(list(latest))
        ],
        answered=[DayCountOut(day=day, count=counts.get(day, 0)) for day in days],
        due=[
            DayCountOut(
                day=day,
                count=sum(due <= day if day == today else due == day for due in dues),
            )
            for day in ahead
        ],
    )


class DeletedOut(BaseModel):
    # What went: answers, their grades, the review history and schedule, ratings, and mock
    # interviews with their follow-ups
    attempts: int
    grades: int
    reviews: int
    cards: int
    ratings: int
    interviews: int


async def remove_practice(session: AsyncSession, user_id: int) -> DeletedOut:
    """Deletes a user's practice, in the session's transaction, and says what went."""
    attempts = select(Attempt.id).where(Attempt.user_id == user_id)
    interview_ids = list(
        await session.scalars(select(Interview.id).where(Interview.user_id == user_id))
    )
    await places.forget(session, interview_ids)
    counts = {}
    for name, rows in (
        ("interviews", delete(Interview).where(Interview.user_id == user_id)),
        ("ratings", delete(Rating).where(Rating.user_id == user_id)),
        ("reviews", delete(Review).where(Review.user_id == user_id)),
        ("grades", delete(Grade).where(Grade.attempt_id.in_(attempts))),
        ("attempts", delete(Attempt).where(Attempt.user_id == user_id)),
        ("cards", delete(Card).where(Card.user_id == user_id)),
    ):
        counts[name] = (await session.execute(rows)).rowcount
    return DeletedOut(**counts)


@router.delete("/practice")
async def delete_practice(session: SessionDep, user_id: UserDep) -> DeletedOut:
    """Delete your practice: every answer with its grades, the review history and schedule
    they made, your ratings, and your mock interviews. The library stays, and so does the count
    of grades the daily limits keep (`app.grading.limits`): deleting doesn't give back the
    day's grades."""
    deleted = await remove_practice(session, user_id)
    await session.commit()
    return deleted

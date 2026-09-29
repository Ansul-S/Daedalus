"""Daily limits on grading.

Every visitor grades out of the same free tiers: Groq's Qwen grades about 110 answers a day,
and one visitor answering fifty questions would spend half of it. So each user has so many
grades a practice day (`DAILY_GRADES_PER_USER`), and everyone together so many over the last
24 hours (`DAILY_GRADES`), set below what the providers allow. A user's own count starts
again when the practice day does, at 04:00 in `PRACTICE_TIMEZONE`. The count in all looks
back 24 hours, as the providers' own allowances and the pacer do: counted from a fixed hour,
a busy hour either side of it could spend two days' worth in one.

What counts is a grade a model replied to: every successful grade, and a failed one that got
a reply, since the reply spent tokens. A grade that failed without any reply (every provider
out of quota, unreachable or refusing) doesn't count, and neither does a request refused here.

Grades are counted in `grade_requests`, not in `grades`. A slot is taken there before the
model is asked, under a lock, so requests sent together can't all pass on the same count; it
outlives its grade, so deleting practice doesn't give the day's grades back; and it records
what each grade cost by model, which the pacer starts from (`app.grading.grader.spent_today`).
A slot whose grade never finished stays counted, which errs towards stopping early.
"""

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from math import inf
from typing import Literal

from pydantic_ai.usage import RunUsage
from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models import Grade, GradeRequest
from app.scheduling.schedule import DAY_STARTS, practice_day

Scope = Literal["per_user", "in_all"]

# How far back the limit in all looks, and how long a row is kept for anything to look back at
WINDOW = timedelta(days=1)
KEPT = timedelta(days=2)
# Held while a slot is counted and taken; a number no other lock in the app uses
LOCK = 6_110_411


@dataclass(frozen=True)
class Limit:
    scope: Scope
    allowed: int
    used: int
    # When a grade is allowed again while none is; None while one is, and for a limit of 0
    again_at: datetime | None

    @property
    def left(self) -> int:
        return max(0, self.allowed - self.used)


@dataclass(frozen=True)
class Allowance:
    per_user: Limit | None
    in_all: Limit | None

    @property
    def limits(self) -> list[Limit]:
        return [limit for limit in (self.per_user, self.in_all) if limit is not None]

    def refusal(self) -> Limit | None:
        """The limit a grade is refused by, if any: of two that are both used up, the one that
        allows a grade again later, since that is when the next one can be asked for. A limit
        of 0 never does."""
        spent = [limit for limit in self.limits if limit.left == 0]
        if not spent:
            return None
        return max(spent, key=lambda limit: limit.again_at.timestamp() if limit.again_at else inf)


class LimitReached(Exception):
    """A grade asked for past a daily limit."""

    def __init__(self, limit: Limit, message: str) -> None:
        super().__init__(message)
        self.limit = limit
        self.message = message


def day_bounds(now: datetime, settings: Settings) -> tuple[datetime, datetime]:
    """When the practice day `now` falls in starts, and when the next one does."""
    zone = settings.practice_zone
    day = practice_day(now, zone)
    start = datetime.combine(day, time(), zone) + DAY_STARTS
    return start, datetime.combine(day + timedelta(days=1), time(), zone) + DAY_STARTS


async def _per_user(
    session: AsyncSession, settings: Settings, user_id: int, now: datetime
) -> Limit | None:
    allowed = settings.daily_grades_per_user
    if allowed is None:
        return None
    start, next_start = day_bounds(now, settings)
    used = await session.scalar(
        select(func.count())
        .select_from(GradeRequest)
        .where(GradeRequest.user_id == user_id, GradeRequest.created_at >= start)
    )
    used = used or 0
    again_at = next_start if used >= allowed and allowed > 0 else None
    return Limit("per_user", allowed, used, again_at)


async def _in_all(session: AsyncSession, settings: Settings, now: datetime) -> Limit | None:
    allowed = settings.daily_grades
    if allowed is None:
        return None
    since = GradeRequest.created_at > now - WINDOW
    used = await session.scalar(select(func.count()).select_from(GradeRequest).where(since)) or 0
    again_at = None
    if used >= allowed and allowed > 0:
        # One more is allowed once enough of the oldest have left the window.
        oldest = await session.scalar(
            select(GradeRequest.created_at)
            .where(since)
            .order_by(GradeRequest.created_at, GradeRequest.id)
            .offset(used - allowed)
            .limit(1)
        )
        again_at = oldest + WINDOW if oldest is not None else None
    return Limit("in_all", allowed, used, again_at)


async def allowance(
    session: AsyncSession, settings: Settings, user_id: int, now: datetime
) -> Allowance:
    """What the limits leave the user at `now`."""
    return Allowance(
        await _per_user(session, settings, user_id, now), await _in_all(session, settings, now)
    )


def refusal_message(limit: Limit, settings: Settings) -> str:
    if limit.again_at is None:
        return "Grading is switched off here for now."
    again = f"{limit.again_at.astimezone(settings.practice_zone):%H:%M %Z}"
    if limit.scope == "per_user":
        return f"You have used today's {limit.allowed} grades; more from {again}."
    return (
        f"All {limit.allowed} grades the last 24 hours allow have been used, by everyone "
        f"together; the next is free at {again}."
    )


async def take(
    session: AsyncSession, settings: Settings, user_id: int, now: datetime
) -> GradeRequest:
    """Take a slot for one grade, or raise `LimitReached`. The slot is counted from the moment
    the caller commits; until then the lock holds back anyone else's."""
    await session.execute(text("SELECT pg_advisory_xact_lock(:lock)"), {"lock": LOCK})
    await session.execute(delete(GradeRequest).where(GradeRequest.created_at < now - KEPT))
    refused = (await allowance(session, settings, user_id, now)).refusal()
    if refused is not None:
        raise LimitReached(refused, refusal_message(refused, settings))
    slot = GradeRequest(user_id=user_id, created_at=now)
    session.add(slot)
    await session.flush()
    return slot


def usage_of(usage: RunUsage) -> dict[str, int]:
    return {
        "requests": usage.requests,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }


async def charge(
    session: AsyncSession,
    slot: GradeRequest | None,
    user_id: int,
    grade: Grade,
    model: str | None,
    usage: RunUsage,
) -> None:
    """Write down what a grade cost, in its slot, or in a row of its own when none was taken.
    A grade no model replied to cost nothing: its slot is given back."""
    if usage.requests == 0:
        if slot is not None:
            await session.delete(slot)
        return
    if slot is None:
        slot = GradeRequest(user_id=user_id)
        session.add(slot)
    slot.grade_id = grade.id
    slot.model = model
    slot.requests = usage.requests
    slot.input_tokens = usage.input_tokens
    slot.output_tokens = usage.output_tokens
    await session.flush()

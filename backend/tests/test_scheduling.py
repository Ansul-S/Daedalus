"""The review schedule: the rating a grade earns, practice days, the intervals py-fsrs gives
with the chosen settings, how well questions and topics are known, and which question
comes next."""

import asyncio
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import fsrs
import pytest

from app.db.models import Attempt, Card, Grade, Question, Review
from app.scheduling.mastery import Standing, standings, topic_mastery
from app.scheduling.picker import choose, next_question
from app.scheduling.schedule import (
    practice_day,
    rating_for,
    record_review,
    retrievability,
    review,
    scheduler,
)

AGAIN, HARD, GOOD, EASY = fsrs.Rating.Again, fsrs.Rating.Hard, fsrs.Rating.Good, fsrs.Rating.Easy
DAY = date(2026, 9, 23)
# Fuzzing moves long intervals at random; switched off, the intervals can be stated exactly.
FIXED = scheduler(fuzz=False)


@pytest.mark.parametrize(
    ("score", "contradicted", "rating"),
    [
        (0.0, 0, AGAIN),
        (0.3999, 0, AGAIN),
        (0.4, 0, HARD),
        (0.6999, 0, HARD),
        (0.7, 0, GOOD),
        (0.8999, 0, GOOD),
        (0.9, 0, EASY),
        (1.0, 0, EASY),
        # A claim the sources contradict makes it Again, whatever the score.
        (1.0, 1, AGAIN),
        (0.85, 2, AGAIN),
    ],
)
def test_a_score_earns_a_rating(score: float, contradicted: int, rating: fsrs.Rating) -> None:
    assert rating_for(score, contradicted) == rating


@pytest.mark.parametrize(
    ("when", "zone", "day"),
    [
        ("2026-09-23T03:59:00+00:00", "UTC", date(2026, 9, 22)),
        ("2026-09-23T04:00:00+00:00", "UTC", date(2026, 9, 23)),
        ("2026-09-23T23:59:00+00:00", "UTC", date(2026, 9, 23)),
        # 03:50 in India, a session that ran past midnight: still the evening's day
        ("2026-09-22T22:20:00+00:00", "Asia/Kolkata", date(2026, 9, 22)),
        ("2026-09-22T22:40:00+00:00", "Asia/Kolkata", date(2026, 9, 23)),
    ],
)
def test_a_practice_day_starts_at_four_in_the_morning(when: str, zone: str, day: date) -> None:
    assert practice_day(datetime.fromisoformat(when), ZoneInfo(zone)) == day


def gaps(ratings: list[fsrs.Rating]) -> list[int]:
    """The days between reviews when every answer is given on the day it is due."""
    card, day, found = fsrs.Card(card_id=1), DAY, []
    for rating in ratings:
        card = review(card, rating, day, FIXED)
        found.append((card.due.date() - day).days)
        day = card.due.date()
    return found


def test_a_first_answer_comes_back_in_days_not_minutes() -> None:
    assert [gaps([rating]) for rating in (AGAIN, HARD, GOOD, EASY)] == [[1], [1], [2], [8]]


def test_no_practised_question_waits_more_than_thirty_days() -> None:
    assert gaps([EASY] * 5) == [8, 30, 30, 30, 30]
    assert gaps([GOOD] * 5) == [2, 11, 30, 30, 30]


def test_a_question_forgotten_after_it_was_known_comes_back_within_days() -> None:
    assert gaps([GOOD, GOOD, GOOD, AGAIN, GOOD]) == [2, 11, 30, 3, 8]


def test_a_partial_answer_every_time_moves_the_question_on_slowly() -> None:
    assert gaps([HARD] * 5) == [1, 3, 7, 12, 18]


def test_the_hour_of_an_answer_does_not_change_the_schedule() -> None:
    """Answered in the evening, then 36 hours later in the morning: two practice days apart.
    Counted in 24-hour spans, py-fsrs would see one day and wait 7 days instead of 11."""
    evening = datetime(2026, 9, 23, 21, tzinfo=UTC)
    morning = datetime(2026, 9, 25, 9, tzinfo=UTC)

    card = review(fsrs.Card(card_id=1), GOOD, practice_day(evening, UTC), FIXED)
    card = review(card, GOOD, practice_day(morning, UTC), FIXED)

    assert card.due.date() - date(2026, 9, 25) == timedelta(days=11)


def test_recall_fades_from_the_day_of_the_answer() -> None:
    state = review(fsrs.Card(card_id=1), GOOD, DAY, FIXED).to_dict()

    recalls = [retrievability(state, DAY + timedelta(days=n)) for n in (0, 2, 10)]

    assert recalls[0] == 1.0
    # Due on the day recall has fallen to 90%, the retention the scheduler aims for
    assert recalls[1] == pytest.approx(0.91, abs=0.01)
    assert recalls[2] < recalls[1]


def standing(
    question_id: int,
    topic_id: int | None = None,
    *,
    difficulty: int = 3,
    score: float | None = None,
    due: date | None = None,
    recall: float = 0.0,
) -> Standing:
    """A question as the picker sees it; one with a score has been practised."""
    state = {"card_id": question_id} if score is not None else None
    return Standing(question_id, topic_id, difficulty, score, state, due, recall)


def test_mastery_is_the_latest_score_as_far_as_it_is_still_recalled() -> None:
    questions = [
        standing(1, 7, score=0.8, due=DAY, recall=0.5),
        standing(2, 7),
        standing(3, None, score=1.0, due=DAY, recall=1.0),
    ]

    assert [question.mastery for question in questions] == [0.4, 0.0, 1.0]
    assert topic_mastery(questions) == {7: 0.2, None: 1.0}


def test_the_most_overdue_question_comes_first() -> None:
    questions = [
        standing(1, score=0.5, due=DAY, recall=0.9),
        standing(2, score=0.9, due=DAY - timedelta(days=3), recall=0.8),
        standing(3),
    ]

    pick = choose(questions, DAY)

    assert pick is not None
    assert (pick.question_id, pick.reason, pick.due) == (2, "due", DAY - timedelta(days=3))
    assert (pick.due_count, pick.new_count) == (2, 1)


def test_with_nothing_due_a_new_question_comes_from_the_weakest_topic() -> None:
    questions = [
        # Topic 1 is half known, topic 2 not at all
        standing(1, 1, score=1.0, due=DAY + timedelta(days=5), recall=0.9),
        standing(2, 1),
        standing(3, 2, difficulty=3),
        standing(4, 2, difficulty=2),
    ]

    pick = choose(questions, DAY)

    assert pick is not None
    # The easier of topic 2's two questions
    assert (pick.question_id, pick.reason, pick.topic_mastery) == (4, "new", 0.0)


def test_among_equally_weak_topics_the_one_practised_least_comes_first() -> None:
    """A failed answer leaves its topic as weak as an untouched one; the untouched one comes
    first, so a newcomer meets every topic in turn."""
    questions = [
        standing(1, 1, score=0.0, due=DAY + timedelta(days=1), recall=0.95),
        standing(2, 1),
        standing(3, 2),
        standing(4, None),
    ]

    pick = choose(questions, DAY)

    assert pick is not None and pick.question_id == 3


def test_with_nothing_due_or_new_it_practises_ahead_on_the_likeliest_forgotten() -> None:
    later = DAY + timedelta(days=4)
    questions = [
        standing(1, score=1.0, due=later, recall=0.95),
        standing(2, score=1.0, due=later, recall=0.8),
        standing(3, score=1.0, due=later, recall=0.9),
    ]

    pick = choose(questions, DAY)

    assert pick is not None
    assert (pick.question_id, pick.reason, pick.due_count, pick.new_count) == (2, "ahead", 0, 0)


def test_an_empty_library_has_nothing_to_pick() -> None:
    assert choose([], DAY) is None


async def add_question(sessions, status: str = "accepted", topic_id: int | None = None) -> int:
    async with sessions() as session, session.begin():
        question = Question(
            text="Why is attention scaled?",
            reference_answer="To keep the softmax gradients large.",
            key_points=[],
            style="why_how",
            difficulty=3,
            status=status,
            topic_id=topic_id,
            generator_model="openai/gpt-oss-120b",
            prompt_version="generate-v3",
        )
        session.add(question)
        await session.flush()
        return question.id


async def answer(
    sessions, question_id: int, score: float, on: date, *, contradicted: int = 0
) -> Review | None:
    """An answer given at noon on a practice day, graded, and recorded in the schedule."""
    async with sessions() as session, session.begin():
        attempt = Attempt(
            question_id=question_id,
            answer="Because large dot products flatten the gradients.",
            created_at=datetime.combine(on, time(12), tzinfo=UTC),
        )
        session.add(attempt)
        await session.flush()
        grade = Grade(
            attempt_id=attempt.id,
            status="graded",
            grader_model="qwen/qwen3.8-27b",
            prompt_version="grade-v1",
            clarity=4,
            coverage=score,
            contradicted=contradicted,
            score=score,
        )
        session.add(grade)
        await session.flush()
        return await record_review(session, grade, UTC)


def test_the_library_stands_on_each_question_s_latest_answer(sessions) -> None:
    async def scenario():
        practised = await add_question(sessions)
        new = await add_question(sessions)
        retired = await add_question(sessions, status="retired")
        first = await answer(sessions, practised, 0.5, DAY)
        second = await answer(sessions, practised, 1.0, DAY + timedelta(days=1))
        await answer(sessions, retired, 1.0, DAY)
        async with sessions() as session:
            found = await standings(session, DAY + timedelta(days=1))
        return practised, new, first, second, found

    practised, new, first, second, found = asyncio.run(scenario())

    assert first is not None and second is not None
    assert (first.rating, first.day, first.due) == (2, DAY, DAY + timedelta(days=1))
    assert [question.question_id for question in found] == [practised, new]
    known, unknown = found
    # The latest answer counts, and it was given that same day
    assert (known.score, known.due, known.retrievability) == (1.0, second.due, 1.0)
    assert (unknown.score, unknown.state, unknown.retrievability, unknown.mastery) == (
        None,
        None,
        0.0,
        0.0,
    )


def test_an_answer_counts_once_and_only_when_graded(sessions) -> None:
    async def scenario():
        question_id = await add_question(sessions)
        async with sessions() as session, session.begin():
            attempt = Attempt(question_id=question_id, answer="Something.")
            session.add(attempt)
            await session.flush()
            failed = Grade(
                attempt_id=attempt.id, status="failed", error="429", prompt_version="grade-v1"
            )
            graded = [
                Grade(
                    attempt_id=attempt.id,
                    status="graded",
                    grader_model="qwen/qwen3.8-27b",
                    prompt_version="grade-v1",
                    clarity=4,
                    coverage=1.0,
                    contradicted=0,
                    score=1.0,
                )
                for _ in range(2)
            ]
            session.add_all([failed, *graded])
            await session.flush()
            return [
                await record_review(session, grade, UTC) for grade in (failed, *graded)
            ], await session.get(Card, question_id)

    (after_failed, after_first, after_second), card = asyncio.run(scenario())

    assert after_failed is None
    assert after_first is not None and after_first.rating == 4
    # Graded again: the schedule already heard about this answer
    assert after_second is None
    assert card is not None and card.due == after_first.due


def test_a_contradicted_claim_brings_a_known_question_back_soon(sessions) -> None:
    async def scenario():
        question_id = await add_question(sessions)
        day, reviews = DAY, []
        for score, contradicted in ((1.0, 0), (1.0, 0), (0.85, 1)):
            reviewed = await answer(sessions, question_id, score, day, contradicted=contradicted)
            assert reviewed is not None
            reviews.append(reviewed)
            day = reviewed.due
        return reviews

    easy, again_easy, contradicted = asyncio.run(scenario())

    assert [easy.rating, again_easy.rating, contradicted.rating] == [4, 4, 1]
    # Well known by then: without the contradiction it would wait weeks
    assert (contradicted.due - contradicted.day).days <= 5


def test_a_retired_question_is_never_picked(sessions) -> None:
    async def scenario():
        retired = await add_question(sessions, status="retired")
        await answer(sessions, retired, 0.0, DAY - timedelta(days=5))
        kept = await add_question(sessions)
        async with sessions() as session:
            pick = await next_question(session, DAY)
        return kept, pick

    kept, pick = asyncio.run(scenario())

    # The retired one is long overdue, but only the library is practised.
    assert pick is not None and (pick.question_id, pick.reason) == (kept, "new")

"""What practice earns: XP and its parts, levels, the streak, and when each coin is minted."""

import asyncio
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest
from test_scheduling import add_question

from app.db.models import Attempt, Grade
from app.scheduling.progress import (
    Answer,
    LibraryQuestion,
    Streak,
    level_for,
    level_start,
    load,
    progress,
    streak_on,
    walk,
    whole,
)
from app.scheduling.schedule import rating_for, record_review

DAY = date(2026, 9, 1)
EARLIER = datetime(2026, 8, 1, tzinfo=UTC)


def answer(
    question_id: int,
    day: date,
    score: float = 0.5,
    *,
    due: date | None = None,
    contradicted: int = 0,
    seconds: float | None = None,
    time_limit: int | None = None,
) -> Answer:
    """A graded answer given at noon on a practice day, due again the day after unless said."""
    return Answer(
        review_id=0,
        attempt_id=0,
        question_id=question_id,
        day=day,
        due=due or day + timedelta(days=1),
        rating=int(rating_for(score, contradicted)),
        score=score,
        contradicted=contradicted,
        seconds=seconds,
        time_limit=time_limit,
        at=datetime.combine(day, time(12), tzinfo=UTC),
    )


def question(
    question_id: int,
    topic_id: int | None = None,
    documents: tuple[int, ...] = (1,),
    *,
    created: datetime = EARLIER,
    accepted: bool = True,
) -> LibraryQuestion:
    return LibraryQuestion(question_id, topic_id, accepted, created, frozenset(documents))


def minted_at(answers: list[Answer], questions: list[LibraryQuestion], coin: str) -> int | None:
    """Which answer, counting from 0, minted a coin; None when none did."""
    for index, step in enumerate(walk(answers, questions).steps):
        if coin in [minted.id for minted in step.minted]:
            return index
    return None


def test_an_answer_earns_ten_times_its_score_and_five_for_answering() -> None:
    (step,) = walk([answer(1, DAY, 0.5)], [question(1)]).steps

    assert (step.xp.score, step.xp.answered, step.xp.due, step.xp.interview) == (5, 5, 0, 0)
    assert (step.xp.streak, step.xp.total, step.total) == (1, 11, 11)


@pytest.mark.parametrize(("value", "rounded"), [(3.5, 4), (3.4999, 3), (4.5, 5), (0.0, 0)])
def test_xp_rounds_half_up(value: float, rounded: int) -> None:
    assert whole(value) == rounded


def test_a_score_of_0_35_earns_4_despite_floating_point() -> None:
    (step,) = walk([answer(1, DAY, 0.35)], [question(1)]).steps

    assert step.xp.score == 4


def test_a_question_due_for_review_earns_half_as_much_again_even_when_late() -> None:
    later = DAY + timedelta(days=10)
    steps = walk(
        [
            answer(1, DAY, 1.0, due=DAY + timedelta(days=2)),
            answer(2, DAY, 1.0, due=DAY + timedelta(days=2)),
            answer(3, DAY, 1.0, due=DAY + timedelta(days=5)),
            # On the day it is due, and eight days late
            answer(1, DAY + timedelta(days=2), 1.0),
            answer(2, later, 1.0),
            # Practising ahead: not due yet
            answer(3, DAY + timedelta(days=2), 1.0),
        ],
        [question(1), question(2), question(3)],
    ).steps

    # (10 + 5) / 2, rounded half up
    assert [step.xp.due for step in steps] == [0, 0, 0, 8, 8, 0]


@pytest.mark.parametrize(
    ("seconds", "time_limit", "earned"),
    [(170.0, 180, 5), (180.0, 180, 5), (200.0, 180, 0), (170.0, None, 0), (None, 180, 0)],
)
def test_interview_mode_earns_five_inside_the_limit(
    seconds: float | None, time_limit: int | None, earned: int
) -> None:
    (step,) = walk([answer(1, DAY, seconds=seconds, time_limit=time_limit)], [question(1)]).steps

    assert step.xp.interview == earned


def test_the_streak_adds_its_length_that_day_up_to_ten() -> None:
    answers = [answer(1, DAY + timedelta(days=offset)) for offset in range(12)]
    # A second answer the same day adds the same again; after a day off it starts over
    answers += [answer(2, DAY + timedelta(days=11)), answer(2, DAY + timedelta(days=13))]

    steps = walk(answers, [question(1), question(2)]).steps

    assert [step.xp.streak for step in steps] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1]


def test_levels_start_at_fifty_times_n_times_n_minus_one() -> None:
    assert [level_start(n) for n in range(1, 8)] == [0, 100, 300, 600, 1000, 1500, 2100]
    assert level_for(0).name == "Apprentice"
    assert (level_for(99).number, level_for(100).number) == (1, 2)
    assert (level_for(99).next_start, level_for(99).next_name) == (100, "Journeyman")
    assert [level_for(level_start(n)).name for n in range(3, 8)] == [
        "Craftsman",
        "Inventor",
        "Architect",
        "Master Builder",
        "Daedalus",
    ]
    top = level_for(5000)
    assert (top.number, top.name, top.start, top.next_start, top.next_name) == (
        7,
        "Daedalus",
        2100,
        None,
        None,
    )


def test_the_answer_that_crosses_a_threshold_is_a_level_up() -> None:
    # 16 XP each: 10 for a perfect score, 5 for answering, 1 for the streak's first day
    steps = walk([answer(1, DAY, 1.0) for _ in range(7)], [question(1)]).steps

    assert [step.total for step in steps] == [16, 32, 48, 64, 80, 96, 112]
    assert [step.level_up for step in steps] == [False] * 6 + [True]
    assert steps[-1].level.name == "Journeyman"


def test_the_streak_holds_until_the_day_ends() -> None:
    days = {DAY - timedelta(days=2), DAY - timedelta(days=1)}

    assert streak_on(days, DAY) == Streak(days=2, today=False, best=2)
    assert (streak_on(days | {DAY}, DAY).days, streak_on(days | {DAY}, DAY).today) == (3, True)
    # A whole day without an answer breaks it; the best run is kept
    assert (streak_on(days, DAY + timedelta(days=1)).days, streak_on(days, DAY).best) == (0, 2)
    assert streak_on(set(), DAY).days == 0


def test_the_first_graded_answer_mints_the_first_thread() -> None:
    assert minted_at([answer(1, DAY), answer(1, DAY)], [question(1)], "first-thread") == 0


def test_seven_days_in_a_row_mint_theseus() -> None:
    week = [answer(1, DAY + timedelta(days=offset)) for offset in range(7)]
    broken = [answer(1, DAY + timedelta(days=offset)) for offset in [0, 1, 2, 4, 5, 6, 7]]

    assert minted_at(week, [question(1)], "theseus") == 6
    assert minted_at(broken, [question(1)], "theseus") is None


def test_a_question_once_failed_then_answered_well_mints_the_minotaur_slayer() -> None:
    answers = [
        answer(2, DAY, 0.95),
        answer(1, DAY, 0.3),
        answer(1, DAY + timedelta(days=1), 0.6),
        answer(1, DAY + timedelta(days=2), 0.9),
    ]

    assert minted_at(answers, [question(1), question(2)], "minotaur-slayer") == 3


def test_a_question_from_every_source_mints_the_cartographer() -> None:
    questions = [
        question(1, documents=(1,)),
        question(2, documents=(1, 2)),
        question(3, documents=(3,)),
    ]

    assert minted_at([answer(1, DAY), answer(2, DAY)], questions, "cartographer") is None
    assert minted_at([answer(2, DAY), answer(3, DAY)], questions, "cartographer") == 1


def test_five_strong_topics_mint_knossos() -> None:
    questions = [question(topic, topic) for topic in range(1, 7)] + [question(7, 6)]
    answers = [answer(topic, DAY, 1.0) for topic in range(1, 7)]

    # Topic 6 has an unanswered question too, so it is only half known.
    assert minted_at(answers[:5], questions, "knossos") == 4
    assert minted_at(answers[1:], questions, "knossos") is None


def test_recall_fading_counts_against_knossos() -> None:
    questions = [question(topic, topic) for topic in range(1, 6)]
    # Two months on, the first four topics are far from certain to be recalled
    answers = [answer(topic, DAY, 1.0) for topic in range(1, 5)]
    answers.append(answer(5, DAY + timedelta(days=60), 1.0))

    assert minted_at(answers, questions, "knossos") is None


def test_an_easy_answer_inside_the_interview_limit_mints_hermes() -> None:
    inside = answer(1, DAY, 0.95, seconds=150.0, time_limit=180)
    over = answer(1, DAY, 0.95, seconds=200.0, time_limit=180)
    unhurried = answer(1, DAY, 0.95)

    assert minted_at([over, unhurried, inside], [question(1)], "hermes") == 2


def test_clearing_every_due_review_in_a_day_mints_ariadne() -> None:
    questions = [question(n) for n in range(1, 5)]
    first = [answer(n, DAY) for n in range(1, 4)]
    tomorrow = DAY + timedelta(days=1)
    cleared = [answer(n, tomorrow) for n in range(1, 4)]

    assert minted_at(first + cleared, questions, "ariadne") == 5
    # Two due are not enough
    assert minted_at(first[:2] + cleared[:2], questions, "ariadne") is None
    # One of the three left for later
    assert minted_at(first + cleared[:2] + [answer(4, tomorrow)], questions, "ariadne") is None


def test_two_contradicted_claims_in_one_answer_mint_icarus() -> None:
    answers = [answer(1, DAY, 0.0, contradicted=1), answer(1, DAY, 0.0, contradicted=2)]

    assert minted_at(answers, [question(1)], "icarus") == 1


def test_every_question_answered_once_mints_the_labyrinth_walker() -> None:
    questions = [question(1), question(2)]

    assert (
        minted_at([answer(1, DAY), answer(1, DAY), answer(2, DAY)], questions, "labyrinth-walker")
        == 2
    )


def test_reaching_the_top_level_mints_daedalus() -> None:
    answers = [answer(1, DAY + timedelta(days=offset), 1.0) for offset in range(120)]

    steps = walk(answers, [question(1)]).steps
    index = minted_at(answers, [question(1)], "daedalus")

    assert index is not None
    assert steps[index].total >= 2100 > steps[index - 1].total
    assert steps[index].level.name == "Daedalus"


def test_a_question_added_later_never_takes_a_coin_back() -> None:
    answers = [answer(1, DAY)]
    later = datetime.combine(DAY + timedelta(days=1), time(), tzinfo=UTC)
    questions = [question(1), question(2, documents=(2,), created=later)]

    ledger = walk(answers, questions)

    assert {coin.id for coin in ledger.steps[0].minted} >= {"labyrinth-walker", "cartographer"}


def test_retired_questions_are_left_out_of_the_library() -> None:
    questions = [question(1), question(2, accepted=False)]

    assert minted_at([answer(1, DAY)], questions, "labyrinth-walker") == 0


def test_progress_counts_what_is_left_of_each_coin_still_to_earn() -> None:
    questions = [question(1, 1, (1,)), question(2, 2, (2,))]
    answers = [answer(1, DAY - timedelta(days=1), 1.0), answer(1, DAY, 1.0)]

    found = progress(walk(answers, questions), DAY)
    coins = {state.coin.id: state for state in found.coins}

    # 16 for the first; 25 for the second, due that day (+8) on the streak's second day
    assert (found.xp, found.level.name, found.answers) == (41, "Apprentice", 2)
    assert (found.streak.days, found.streak.today, found.streak.best) == (2, True, 2)
    assert len(found.coins) == 10
    assert coins["first-thread"].minted_on == DAY - timedelta(days=1)
    counts = {
        coin: (state.have, state.need) for coin, state in coins.items() if state.minted_on is None
    }
    assert counts == {
        "theseus": (2, 7),
        "minotaur-slayer": (None, None),
        "cartographer": (1, 2),
        "knossos": (1, 5),
        "hermes": (None, None),
        "ariadne": (None, None),
        "icarus": (None, None),
        "labyrinth-walker": (1, 2),
        "daedalus": (41, 2100),
    }


def test_nothing_is_earned_before_the_first_answer() -> None:
    found = progress(walk([], [question(1)]), DAY)

    assert (found.xp, found.level.number, found.streak.days, found.answers) == (0, 1, 0, 0)
    assert all(state.minted_on is None for state in found.coins)
    assert (found.coins[0].have, found.coins[0].need) == (0, 1)


def test_the_streak_counts_practice_days_that_start_at_four_in_the_morning(sessions) -> None:
    """Two answers in Kolkata on the morning of 25 September, one before 04:00 and one after:
    the first counts for the 24th, so the streak is two days long."""
    kolkata = ZoneInfo("Asia/Kolkata")

    async def scenario() -> list[date]:
        question_id = await add_question(sessions)
        for hour, minute in [(3, 30), (4, 30)]:
            async with sessions() as session, session.begin():
                given = datetime(2026, 9, 25, hour, minute, tzinfo=kolkata)
                attempt = Attempt(question_id=question_id, answer="…", created_at=given)
                session.add(attempt)
                await session.flush()
                grade = Grade(
                    attempt_id=attempt.id,
                    status="graded",
                    grader_model="qwen/qwen3.8-27b",
                    prompt_version="grade-v1",
                    clarity=4,
                    coverage=0.5,
                    contradicted=0,
                    score=0.5,
                )
                session.add(grade)
                await session.flush()
                await record_review(session, grade, kolkata)
        async with sessions() as session:
            answers, questions = await load(session)
        steps = walk(answers, questions).steps
        assert [step.xp.streak for step in steps] == [1, 2]
        return [step.answer.day for step in steps]

    assert asyncio.run(scenario()) == [date(2026, 9, 24), date(2026, 9, 25)]

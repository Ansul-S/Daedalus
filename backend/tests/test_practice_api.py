"""Practising through the API: the next question to answer, and how answering it moves the
question in the schedule."""

import asyncio
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select, update
from test_grading_api import ANSWER, add_question, grading, refusing, use_grader

from app.db.models import Card, Review
from app.scheduling.schedule import practice_day


def today():
    # The test settings count practice days in UTC.
    return practice_day(datetime.now(UTC), UTC)


def reviews(sessions) -> int:
    async def count() -> int:
        async with sessions() as session:
            return await session.scalar(select(func.count()).select_from(Review)) or 0

    return asyncio.run(count())


def card_due(sessions, question_id: int):
    async def due():
        async with sessions() as session:
            card = await session.get(Card, question_id)
            return card.due if card else None

    return asyncio.run(due())


def test_there_is_nothing_to_practise_without_questions(client) -> None:
    response = client.get("/practice/next")

    assert response.status_code == 404
    assert "generate" in response.json()["detail"]


def test_the_next_question_comes_without_its_answer(client, sessions, corpus) -> None:
    question_id = asyncio.run(add_question(sessions, corpus.scaling))
    asyncio.run(add_question(sessions, corpus.scaling, "retired"))

    response = client.get("/practice/next")

    assert response.status_code == 200
    practice = response.json()
    question = practice["question"]
    assert question["id"] == question_id
    assert question["citations"][0].startswith("Attention Is All You Need")
    assert "reference_answer" not in question and "key_points" not in question
    assert (practice["reason"], practice["why"], practice["due"]) == ("new", "new", None)
    assert (practice["recall"], practice["due_count"], practice["new_count"]) == (0.0, 0, 1)


def test_an_answer_reschedules_its_question_and_the_next_one_moves_on(
    client, sessions, corpus
) -> None:
    first, second = (asyncio.run(add_question(sessions, corpus.scaling)) for _ in range(2))
    use_grader(grading(corpus.scaling))
    assert client.get("/practice/next").json()["question"]["id"] == first

    attempt = client.post(f"/questions/{first}/attempts", json={"answer": ANSWER}).json()

    # Scored 0.67: a hard recall, back tomorrow
    tomorrow = today() + timedelta(days=1)
    assert attempt["review"] == {
        "rating": "hard",
        "day": today().isoformat(),
        "due": tomorrow.isoformat(),
        "interval": 1,
    }
    assert card_due(sessions, first) == tomorrow
    following = client.get("/practice/next").json()
    assert (following["question"]["id"], following["reason"]) == (second, "new")
    assert (following["due_count"], following["new_count"]) == (0, 1)


def test_grading_an_answer_again_does_not_count_it_twice(client, sessions, corpus) -> None:
    question_id = asyncio.run(add_question(sessions, corpus.scaling))
    use_grader(grading(corpus.scaling))
    attempt = client.post(f"/questions/{question_id}/attempts", json={"answer": ANSWER}).json()

    again = client.post(f"/attempts/{attempt['id']}/grades").json()

    assert [grade["status"] for grade in again["grades"]] == ["graded", "graded"]
    assert again["review"] == attempt["review"]
    assert reviews(sessions) == 1


def test_a_failed_grade_leaves_the_schedule_alone_until_it_is_graded(
    client, sessions, corpus
) -> None:
    question_id = asyncio.run(add_question(sessions, corpus.scaling))
    use_grader(refusing())
    failed = client.post(f"/questions/{question_id}/attempts", json={"answer": ANSWER}).json()

    assert failed["review"] is None
    assert (reviews(sessions), card_due(sessions, question_id)) == (0, None)
    assert client.get("/practice/next").json()["reason"] == "new"

    use_grader(grading(corpus.scaling))
    graded = client.post(f"/attempts/{failed['id']}/grades").json()

    assert graded["review"]["rating"] == "hard"
    assert reviews(sessions) == 1


def test_a_question_due_for_review_comes_before_new_ones(client, sessions, corpus) -> None:
    practised, _ = (asyncio.run(add_question(sessions, corpus.scaling)) for _ in range(2))
    use_grader(grading(corpus.scaling))
    client.post(f"/questions/{practised}/attempts", json={"answer": ANSWER})

    async def overdue() -> None:
        async with sessions() as session, session.begin():
            await session.execute(
                update(Card)
                .where(Card.question_id == practised)
                .values(due=today() - timedelta(days=1))
            )

    asyncio.run(overdue())

    practice = client.get("/practice/next").json()

    assert (practice["question"]["id"], practice["reason"]) == (practised, "due")
    assert practice["why"] == "due for review, 1 day late"
    assert (practice["due_count"], practice["new_count"]) == (1, 1)

"""Ratings: rating a question or a grade, reading the latest rating back wherever the question
or grade is shown, and narrowing the library down by it."""

import asyncio

import pytest
from sqlalchemy import select
from test_grading_api import ANSWER, add_question, grading, refusing, use_grader

from app.api.ratings import MAX_NOTE_CHARS
from app.db.models import Rating


def kept(sessions) -> list[tuple]:
    """Every rating stored, oldest first: what was rated, the value and the note."""

    async def load() -> list[tuple]:
        async with sessions() as session:
            rows = await session.execute(
                select(Rating.question_id, Rating.grade_id, Rating.value, Rating.note).order_by(
                    Rating.id
                )
            )
            return [tuple(row) for row in rows]

    return asyncio.run(load())


def test_a_question_is_rated_and_its_rating_comes_with_it(client, sessions, corpus) -> None:
    rated, other = (asyncio.run(add_question(sessions, corpus.scaling)) for _ in range(2))

    response = client.post("/ratings", json={"question_id": rated, "value": 1})

    assert response.status_code == 201
    rating = response.json()
    assert [rating[field] for field in ("question_id", "grade_id", "value", "note")] == [
        rated,
        None,
        1,
        None,
    ]
    assert rating["id"] and rating["created_at"]
    assert client.get(f"/questions/{rated}").json()["rating"] == rating
    listed = client.get("/questions").json()["results"]
    assert {question["id"]: question["rating"] for question in listed} == {
        rated: rating,
        other: None,
    }
    # The practice page gets it with the question, so a rated question shows its rating there
    practice = client.get("/practice/next").json()
    assert (practice["question"]["id"], practice["question"]["rating"]) == (rated, rating)


def test_a_poor_rating_keeps_its_note(client, sessions, corpus) -> None:
    question_id = asyncio.run(add_question(sessions, corpus.scaling))
    note = "  The second key point says the first one again.  "

    noted = client.post("/ratings", json={"question_id": question_id, "value": -1, "note": note})
    blank = client.post("/ratings", json={"question_id": question_id, "value": -1, "note": " "})

    assert noted.json()["note"] == "The second key point says the first one again."
    # A blank note says nothing
    assert blank.json()["note"] is None


def test_the_latest_rating_stands_and_every_rating_is_kept(client, sessions, corpus) -> None:
    question_id = asyncio.run(add_question(sessions, corpus.scaling))
    note = "It asks for a number the passage only reports."

    client.post("/ratings", json={"question_id": question_id, "value": 1})
    latest = client.post("/ratings", json={"question_id": question_id, "value": -1, "note": note})

    assert client.get(f"/questions/{question_id}").json()["rating"] == latest.json()
    assert kept(sessions) == [(question_id, None, 1, None), (question_id, None, -1, note)]


def test_a_grade_is_rated_and_its_rating_comes_with_its_attempt(client, sessions, corpus) -> None:
    question_id = asyncio.run(add_question(sessions, corpus.scaling))
    use_grader(grading(corpus.scaling))
    attempt = client.post(f"/questions/{question_id}/attempts", json={"answer": ANSWER}).json()
    [grade] = attempt["grades"]
    note = "The answer never says what the scale is, and k1 was marked covered anyway."

    response = client.post("/ratings", json={"grade_id": grade["id"], "value": -1, "note": note})

    assert response.status_code == 201
    rating = response.json()
    assert [rating[field] for field in ("question_id", "grade_id", "value", "note")] == [
        None,
        grade["id"],
        -1,
        note,
    ]
    assert grade["rating"] is None
    assert client.get(f"/attempts/{attempt['id']}").json()["grades"][0]["rating"] == rating
    listed = client.get(f"/questions/{question_id}/attempts").json()
    assert listed[0]["grades"][0]["rating"] == rating
    # Each grade has its own: a second grade of the same answer starts unrated
    again = client.post(f"/attempts/{attempt['id']}/grades").json()
    assert [grade["rating"] for grade in again["grades"]] == [rating, None]
    # And rating a grade says nothing about its question
    assert client.get(f"/questions/{question_id}").json()["rating"] is None


def test_a_failed_grade_has_no_verdict_to_rate(client, sessions, corpus) -> None:
    question_id = asyncio.run(add_question(sessions, corpus.scaling))
    use_grader(refusing())
    attempt = client.post(f"/questions/{question_id}/attempts", json={"answer": ANSWER}).json()
    [failed] = attempt["grades"]

    response = client.post("/ratings", json={"grade_id": failed["id"], "value": -1})

    assert response.status_code == 409
    assert "grade the answer again" in response.json()["detail"]
    assert kept(sessions) == []


@pytest.mark.parametrize("state", ["rejected", "retired"])
def test_any_question_can_be_rated_whatever_its_status(client, sessions, corpus, state) -> None:
    question_id = asyncio.run(add_question(sessions, corpus.scaling, state))

    # A rejected question rated good is a check that turned down too much
    response = client.post("/ratings", json={"question_id": question_id, "value": 1})

    assert response.status_code == 201
    assert client.get(f"/questions/{question_id}").json()["rating"]["value"] == 1


@pytest.mark.parametrize(
    ("body", "detail"),
    [
        pytest.param({"question_id": 999, "value": 1}, "question not found", id="question"),
        pytest.param({"grade_id": 999, "value": -1}, "grade not found", id="grade"),
    ],
)
def test_only_a_question_or_grade_that_exists_can_be_rated(client, body, detail) -> None:
    response = client.post("/ratings", json=body)

    assert response.status_code == 404
    assert response.json()["detail"] == detail


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"value": 1}, id="nothing rated"),
        pytest.param({"question_id": 1, "grade_id": 1, "value": 1}, id="two things rated"),
        pytest.param({"question_id": 1}, id="no value"),
        pytest.param({"question_id": 1, "value": 0}, id="neither good nor poor"),
        pytest.param({"question_id": 1, "value": 2}, id="better than good"),
        pytest.param({"question_id": 1, "value": "good"}, id="a word"),
        pytest.param(
            {"question_id": 1, "value": -1, "note": "x" * (MAX_NOTE_CHARS + 1)}, id="long note"
        ),
        pytest.param({"question_id": 1, "value": 1, "stars": 5}, id="unknown field"),
    ],
)
def test_a_rating_has_to_be_one_that_can_be_kept(client, body) -> None:
    assert client.post("/ratings", json=body).status_code == 422


def test_the_library_can_be_filtered_by_the_latest_rating(client, sessions, corpus) -> None:
    good, changed, unrated, retired = (
        asyncio.run(add_question(sessions, corpus.scaling, state))
        for state in ("accepted", "accepted", "accepted", "retired")
    )
    for question_id, value in ((good, 1), (changed, 1), (changed, -1), (retired, -1)):
        client.post("/ratings", json={"question_id": question_id, "value": value})
    # A poor grade of an answer to a question is not a rating of the question
    use_grader(grading(corpus.scaling))
    attempt = client.post(f"/questions/{unrated}/attempts", json={"answer": ANSWER}).json()
    client.post("/ratings", json={"grade_id": attempt["grades"][0]["id"], "value": -1})

    def listed(**params) -> list[int]:
        body = client.get("/questions", params=params).json()
        assert body["total"] == len(body["results"])
        return sorted(question["id"] for question in body["results"])

    assert listed(rating="good") == [good]
    # Rated good, then poor: the latest rating is the one that counts
    assert listed(rating="poor") == [changed, retired]
    assert listed(rating="poor", status="accepted") == [changed]
    assert listed(rating="unrated") == [unrated]

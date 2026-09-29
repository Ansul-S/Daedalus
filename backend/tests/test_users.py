"""Users: whose practice is whose. The library is shared; each user's answers, schedule,
progress and ratings are their own. Locally the built-in user owns everything, and in
production nobody is the built-in user."""

import asyncio
import json
from collections.abc import Iterator
from datetime import date, timedelta

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from conftest import BACKEND
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile
from sqlalchemy import delete, select
from sqlalchemy.engine import make_url
from test_grading_api import ANSWER, add_question, grader_output, grading, use_grader
from test_practice_api import add_topic
from test_scheduling import add_question as add_bare_question
from test_scheduling import answer

from app.api.users import local_user, user_or_none
from app.db.models import LOCAL_USER, Card, User
from app.main import app
from app.scheduling.progress import load, walk

DAY = date(2026, 9, 1)


async def add_user(sessions, subject: str) -> int:
    async with sessions() as session, session.begin():
        user = User(provider="test", subject=subject)
        session.add(user)
        await session.flush()
        return user.id


def builtin_user(sessions) -> int:
    async def find() -> int:
        async with sessions() as session:
            return await local_user(session)

    return asyncio.run(find())


def act_as(user_id: int) -> None:
    """Send the next requests as this user. The client fixture clears every override when the
    test ends."""
    app.dependency_overrides[user_or_none] = lambda: user_id


def practised(client, sessions, corpus, embedder) -> tuple[int, int, dict]:
    """Two questions under one topic; the built-in user answers the first and rates it."""
    first, second = (asyncio.run(add_question(sessions, corpus.scaling)) for _ in range(2))
    asyncio.run(add_topic(sessions, embedder, "attention", first, second))
    use_grader(grading(corpus.scaling))
    attempt = client.post(f"/questions/{first}/attempts", json={"answer": ANSWER}).json()
    assert client.post("/ratings", json={"question_id": first, "value": 1}).status_code == 201
    return first, second, attempt


def test_a_new_user_starts_from_nothing_whatever_others_practised(
    client, sessions, corpus, embedder
) -> None:
    first, second, _ = practised(client, sessions, corpus, embedder)
    mine = client.get("/practice/next").json()
    act_as(asyncio.run(add_user(sessions, "someone")))

    theirs = client.get("/practice/next").json()
    progress = client.get("/practice/progress").json()
    stats = client.get("/practice/stats").json()
    labyrinth = client.get("/practice/map").json()

    # The built-in user has moved on to the second question; for them both are still new.
    assert (mine["question"]["id"], mine["new_count"]) == (second, 1)
    assert (theirs["question"]["id"], theirs["reason"], theirs["new_count"]) == (first, "new", 2)
    assert theirs["question"]["rating"] is None
    assert (progress["xp"], progress["answers"], progress["streak"]["days"]) == (0, 0, 0)
    assert not any(coin["minted_on"] for coin in progress["coins"])
    assert stats["scores"] == [] and not any(day["count"] for day in stats["answered"])
    assert [(room["practised"], room["mastery"]) for room in labyrinth["rooms"]] == [(0, 0.0)]
    assert labyrinth["visits"] == []
    assert client.get(f"/questions/{first}/attempts").json() == []
    assert client.get(f"/questions/{first}").json()["rating"] is None
    assert client.get("/questions", params={"rating": "good"}).json()["total"] == 0
    assert client.get("/questions", params={"rating": "unrated"}).json()["total"] == 2


def test_another_user_s_answer_cannot_be_read_graded_again_or_rated(
    client, sessions, corpus, embedder
) -> None:
    _, _, attempt = practised(client, sessions, corpus, embedder)
    [grade] = attempt["grades"]
    act_as(asyncio.run(add_user(sessions, "someone")))

    assert client.get(f"/attempts/{attempt['id']}").status_code == 404
    assert client.post(f"/attempts/{attempt['id']}/grades").status_code == 404
    rated = client.post("/ratings", json={"grade_id": grade["id"], "value": -1})
    assert (rated.status_code, rated.json()["detail"]) == (404, "grade not found")


def grading_everything(chunk_id: int) -> FunctionModel:
    """A grader that finds both key points covered, for a score of 1."""
    output = grader_output(chunk_id)
    output["key_points"][1] = {"id": "k2", "status": "covered", "answer_quote": "scaled"}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(json.dumps(output))])

    return FunctionModel(respond, profile=ModelProfile(supports_json_schema_output=True))


def test_two_users_answering_one_question_keep_schedules_of_their_own(
    client, sessions, corpus, embedder
) -> None:
    first, _, mine = practised(client, sessions, corpus, embedder)
    someone = asyncio.run(add_user(sessions, "someone"))
    act_as(someone)
    use_grader(grading_everything(corpus.scaling))

    theirs = client.post(f"/questions/{first}/attempts", json={"answer": ANSWER}).json()
    their_progress = client.get("/practice/progress").json()
    their_map = client.get("/practice/map").json()
    their_attempts = client.get(f"/questions/{first}/attempts").json()
    app.dependency_overrides.pop(user_or_none)
    my_progress = client.get("/practice/progress").json()
    my_map = client.get("/practice/map").json()
    my_attempts = client.get(f"/questions/{first}/attempts").json()

    async def card_owners() -> set[int]:
        async with sessions() as session:
            return set(await session.scalars(select(Card.user_id).where(Card.question_id == first)))

    assert [attempt["grades"][0]["score"] for attempt in (mine, theirs)] == [0.6667, 1.0]
    # Each answer is the first of its user's history, and the only one in it.
    assert "first-thread" in [coin["id"] for coin in theirs["earned"]["coins"]]
    assert (my_progress["answers"], their_progress["answers"]) == (1, 1)
    assert my_progress["xp"] == mine["earned"]["total_xp"]
    assert their_progress["xp"] == theirs["earned"]["total_xp"]
    # A topic's mastery is the mean over its two questions of each user's own latest score.
    assert my_map["rooms"][0]["mastery"] == pytest.approx(0.3333, abs=0.0001)
    assert their_map["rooms"][0]["mastery"] == 0.5
    assert [attempt["id"] for attempt in my_attempts] == [mine["id"]]
    assert [attempt["id"] for attempt in their_attempts] == [theirs["id"]]
    assert asyncio.run(card_owners()) == {builtin_user(sessions), someone}


def test_each_user_s_history_is_replayed_on_its_own(sessions) -> None:
    async def scenario():
        question_id = await add_bare_question(sessions)
        someone = await add_user(sessions, "someone")
        # Their answers interleave, on the same question and on the same days.
        for offset, (mine, theirs) in enumerate([(0.2, 1.0), (1.0, 0.5), (1.0, 1.0)]):
            day = DAY + timedelta(days=offset)
            await answer(sessions, question_id, mine, day)
            await answer(sessions, question_id, theirs, day, user_id=someone)
        async with sessions() as session:
            return [
                walk(*await load(session, user_id)).steps
                for user_id in (await local_user(session), someone)
            ]

    my_steps, their_steps = asyncio.run(scenario())

    assert [step.answer.score for step in my_steps] == [0.2, 1.0, 1.0]
    assert [step.answer.score for step in their_steps] == [1.0, 0.5, 1.0]
    # A streak of three days each, from answers that were each user's own
    assert [step.streak for step in my_steps] == [step.streak for step in their_steps] == [1, 2, 3]
    # Only the built-in user went from under 0.4 to 0.9 or more on the question.
    mine, theirs = (
        {coin.id for step in steps for coin in step.minted} for steps in (my_steps, their_steps)
    )
    assert "minotaur-slayer" in mine and "minotaur-slayer" not in theirs


@pytest.mark.parametrize(
    "method, path, body",
    [
        ("GET", "/practice/next", None),
        ("GET", "/practice/progress", None),
        ("GET", "/practice/map", None),
        ("GET", "/practice/stats", None),
        ("POST", "/questions/1/attempts", {"answer": ANSWER}),
        ("POST", "/attempts/1/grades", None),
        ("GET", "/attempts/1", None),
        ("GET", "/questions/1/attempts", None),
        ("POST", "/ratings", {"question_id": 1, "value": 1}),
    ],
)
def test_practice_needs_a_signed_in_user_in_production(
    client, settings, method, path, body
) -> None:
    settings.environment = "production"

    response = client.request(method, path, json=body)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_the_library_is_read_in_production_without_anyone_s_ratings(
    client, sessions, corpus, settings
) -> None:
    question_id = asyncio.run(add_question(sessions, corpus.scaling))
    client.post("/ratings", json={"question_id": question_id, "value": 1})
    settings.environment = "production"

    listed = client.get("/questions")

    assert listed.status_code == 200
    assert [question["rating"] for question in listed.json()["results"]] == [None]
    assert client.get(f"/questions/{question_id}").json()["rating"] is None
    assert client.get("/questions", params={"rating": "good"}).json()["total"] == 0
    assert client.get("/questions", params={"rating": "unrated"}).json()["total"] == 1


def test_the_built_in_user_is_added_again_when_it_is_missing(client, sessions) -> None:
    async def users() -> list[tuple[str, str]]:
        async with sessions() as session:
            return list((await session.execute(select(User.provider, User.subject))).tuples())

    async def remove() -> None:
        async with sessions() as session, session.begin():
            await session.execute(delete(User))

    asyncio.run(remove())

    assert client.get("/practice/progress").status_code == 200
    assert client.get("/practice/progress").status_code == 200
    assert asyncio.run(users()) == [LOCAL_USER]


# The migration is walked on a database of its own, so the shared test database stays at the
# latest revision.
MIGRATING = "daedalus_migration_test"

# Practice as it stood before users: two answers to one question, the first graded and
# reviewed, the second's grade failed; the question's card; a rating of the question and one
# of the grade.
BEFORE_USERS = """
INSERT INTO questions (id, text, reference_answer, style, difficulty, generator_model,
                       prompt_version)
VALUES (1, 'Why is attention scaled?', 'To keep the gradients large.', 'why_how', 3,
        'openai/gpt-oss-120b', 'generate-v3');
INSERT INTO attempts (id, question_id, answer) VALUES (1, 1, 'First.'), (2, 1, 'Second.');
INSERT INTO grades (id, attempt_id, status, grader_model, prompt_version, clarity, coverage,
                    contradicted, score, error)
VALUES (1, 1, 'graded', 'qwen/qwen3.8-27b', 'grade-v1', 4, 1.0, 0, 1.0, NULL),
       (2, 2, 'failed', NULL, 'grade-v1', NULL, NULL, NULL, NULL, '429');
INSERT INTO reviews (id, question_id, attempt_id, grade_id, rating, score, day, due)
VALUES (1, 1, 1, 1, 4, 1.0, '2026-09-28', '2026-10-01');
INSERT INTO cards (question_id, state, due) VALUES (1, '{}', '2026-10-01');
INSERT INTO ratings (id, question_id, grade_id, value) VALUES (1, 1, NULL, 1), (2, NULL, 1, -1);
"""

PRACTICE = {"attempts": 2, "grades": 2, "reviews": 1, "cards": 1, "ratings": 2}


@pytest.fixture
def migrating(database_url: str) -> Iterator[tuple[Config, str]]:
    """An empty database, and the Alembic config that migrates it."""
    url = make_url(database_url)
    admin = url.set(drivername="postgresql", database="postgres").render_as_string(
        hide_password=False
    )
    with psycopg.connect(admin, autocommit=True) as connection:
        connection.execute(f"DROP DATABASE IF EXISTS {MIGRATING} WITH (FORCE)")
        connection.execute(f"CREATE DATABASE {MIGRATING}")
    migrated = url.set(database=MIGRATING)
    config = Config(
        toml_file=BACKEND / "pyproject.toml",
        attributes={"database_url": migrated.render_as_string(hide_password=False)},
    )
    yield config, migrated.set(drivername="postgresql").render_as_string(hide_password=False)
    with psycopg.connect(admin, autocommit=True) as connection:
        connection.execute(f"DROP DATABASE IF EXISTS {MIGRATING} WITH (FORCE)")


def counts(connection: psycopg.Connection) -> dict[str, int]:
    return {
        table: connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for table in PRACTICE
    }


def test_the_migration_gives_the_practice_so_far_to_the_built_in_user(migrating) -> None:
    config, url = migrating
    command.upgrade(config, "0008")
    with psycopg.connect(url) as connection:
        connection.execute(BEFORE_USERS)

    command.upgrade(config, "0009")
    with psycopg.connect(url) as connection:
        owners = {
            table: {row[0] for row in connection.execute(f"SELECT user_id FROM {table}")}
            for table in ("attempts", "reviews", "cards", "ratings")
        }
        users = connection.execute("SELECT id, provider, subject FROM users").fetchall()
        after = counts(connection)
        # Someone else can now practise the same question, with a card of their own.
        [(someone,)] = connection.execute(
            "INSERT INTO users (provider, subject) VALUES ('test', 'someone') RETURNING id"
        ).fetchall()
        connection.execute(
            "INSERT INTO attempts (id, user_id, question_id, answer) VALUES (3, %s, 1, 'Mine.')",
            (someone,),
        )
        connection.execute(
            "INSERT INTO cards (user_id, question_id, state, due) VALUES (%s, 1, '{}', %s)",
            (someone, date(2026, 10, 2)),
        )

    [(local, *identity)] = users
    assert tuple(identity) == LOCAL_USER
    assert owners == dict.fromkeys(owners, {local})
    assert after == PRACTICE

    # Going back keeps the built-in user's practice and drops everyone else's.
    command.downgrade(config, "0008")
    with psycopg.connect(url) as connection:
        assert counts(connection) == PRACTICE
        assert connection.execute("SELECT count(*) FROM attempts WHERE id = 3").fetchone() == (0,)

"""Mock interviews: questions asked in turn, a follow-up on what an answer missed, the report,
the daily limits, and the interview's place kept in Postgres from one request to the next."""

import asyncio
import json
import re
from datetime import UTC, date, datetime, timedelta

import httpx
import psycopg
import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langsmith import utils as langsmith_utils
from langsmith.run_helpers import get_tracing_context
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile
from sqlalchemy import func, select, update
from sqlalchemy import text as sql_text
from sqlalchemy.engine import make_url
from test_grading_api import use_grader
from test_limits import limited
from test_practice_api import add_topic
from test_users import add_user

from app.api import interviews as interviews_api
from app.api.interviews import get_writer
from app.db.models import (
    CHECKPOINTS,
    Attempt,
    Grade,
    GradeRequest,
    Interview,
    InterviewTurn,
    Question,
    QuestionSource,
    Review,
)
from app.grading.grader import spent_today
from app.interview import routing
from app.interview.follow_ups import FollowUpWriter, Writing
from app.interview.picking import plan
from app.interview.report import Round, report
from app.llm import fakes
from app.llm.fakes import prompt
from app.main import app
from app.scheduling.mastery import Standing

SCALED = "divide the dot products by the square root of the key dimension"
GRADIENTS = "which keeps the softmax gradients large"
# The scaling passage's one sentence, which the stand-in writer copies as its evidence
SCALING = (
    "We divide the dot products by the square root of the key dimension, "
    "which keeps the softmax gradients large."
)


async def add_library(sessions, corpus, embedder) -> list[int]:
    """Three questions, each under a topic of its own, with two key points resting on its
    passage: the first weighs 2, the second 1."""
    specs = [
        (
            corpus.scaling,
            "attention",
            "Why are the dot products scaled?",
            [(SCALED, "the dot products are scaled"), (GRADIENTS, "it keeps gradients large")],
        ),
        (
            corpus.vanishing,
            "rnn",
            "Why do simple RNNs forget the start of a long sequence?",
            [
                ("Gradients shrink as they flow back", "gradients shrink going back"),
                ("through many time steps", "over many time steps"),
            ],
        ),
        (
            corpus.positions,
            "positions",
            "What do sinusoidal encodings give the model?",
            [
                ("Sinusoidal encodings let the model attend", "they let it attend"),
                ("by relative position", "by relative position"),
            ],
        ),
    ]
    ids = []
    for chunk_id, topic, text, points in specs:
        async with sessions() as session, session.begin():
            question = Question(
                text=text,
                reference_answer="As the passage says.",
                key_points=[
                    {"text": point, "weight": weight, "evidence_quote": quote, "chunk_id": chunk_id}
                    for (quote, point), weight in zip(points, (2, 1), strict=True)
                ],
                style="why_how",
                difficulty=2,
                generator_model="openai/gpt-oss-120b",
                prompt_version="generate-v8",
            )
            session.add(question)
            await session.flush()
            session.add(QuestionSource(question_id=question.id, chunk_id=chunk_id, position=0))
            ids.append(question.id)
        await add_topic(sessions, embedder, topic, ids[-1])
    return ids


def interviewer_grader(*, crash: bool = False) -> FunctionModel:
    """Grades by what the answer says it is: FULL covers every point, GAP the first alone,
    WRONG the first alone with a claim the passage contradicts, MISS none, and FAIL finds the
    grader out of service. With `crash`, grading breaks outright, as a bug would."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if crash:
            raise RuntimeError("the grader broke")
        request = prompt(messages)
        ids = re.findall(r"^- (k\d+): ", request, re.MULTILINE)
        chunk = int(re.findall(r"^\[chunk (\d+)\]", request, re.MULTILINE)[0])
        answer = request.split("<<<answer", 1)[1].rsplit("answer>>>", 1)[0]
        if "FAIL" in answer:
            raise ModelHTTPError(status_code=503, model_name="test-grader", body="unavailable")
        first = "missing" if "MISS" in answer else "covered"
        rest = "covered" if "FULL" in answer else "missing"
        wrong = "WRONG" in answer
        labels = [first] + [rest] * (len(ids) - 1)
        output = {
            "key_points": [
                {"id": id_, "status": status, "answer_quote": "" if status == "missing" else "It"}
                for id_, status in zip(ids, labels, strict=True)
            ],
            "claims": [
                {
                    "claim": "It works the other way round.",
                    "verdict": "contradicted" if wrong else "supported",
                    "chunk_id": chunk,
                    "why": "The passage says otherwise." if wrong else "The passage says so.",
                }
            ],
            "clarity": 4,
            "strengths": [],
            "gaps": [],
            "errors": ["It works the other way round."] if wrong else [],
            "improved_answer": "What the passage says.",
            "follow_up": "And then?",
        }
        return ModelResponse(parts=[TextPart(json.dumps(output))])

    return FunctionModel(
        respond, model_name="test-grader", profile=ModelProfile(supports_json_schema_output=True)
    )


def use_writer(writer: object = "stand-in") -> None:
    """Have follow-ups written by `writer`; by default, the real writer on the stand-in model."""
    if writer == "stand-in":
        writer = FollowUpWriter(fakes.follow_up_writer())
    app.dependency_overrides[get_writer] = lambda: writer


def answer(client, interview: dict, text: str, seconds: float = 30.0):
    return client.post(
        f"/interviews/{interview['id']}/answers",
        json={
            "turn_id": interview["waiting"],
            "answer": text,
            "seconds": seconds,
            "time_limit": 180,
        },
    )


def count(sessions, what) -> int:
    async def counted() -> int:
        async with sessions() as session:
            return await session.scalar(select(func.count()).select_from(what)) or 0

    return asyncio.run(counted())


async def charged(sessions) -> list[tuple[str | None, int]]:
    async with sessions() as session:
        rows = await session.execute(
            select(GradeRequest.model, GradeRequest.requests).order_by(GradeRequest.id)
        )
        return [tuple(row) for row in rows]


async def spent(sessions) -> dict[str, tuple[int, int]]:
    async with sessions() as session:
        return await spent_today(session)


def kept_in_places(sessions, text: str) -> bool:
    """Whether LangGraph's saved places hold `text` anywhere."""

    async def search() -> bool:
        async with sessions() as session:
            found = await session.scalar(
                text_query,
                {"text": f"%{text}%"},
            )
            return bool(found)

    return asyncio.run(search())


text_query = sql_text(
    "SELECT EXISTS (SELECT 1 FROM checkpoint_writes WHERE encode(blob, 'escape') LIKE :text) "
    "OR EXISTS (SELECT 1 FROM checkpoint_blobs WHERE encode(blob, 'escape') LIKE :text) "
    "OR EXISTS (SELECT 1 FROM checkpoints WHERE checkpoint::text LIKE :text "
    "OR metadata::text LIKE :text)"
)


@pytest.fixture
def library(client, sessions, corpus, embedder) -> list[int]:
    use_grader(interviewer_grader())
    use_writer()
    return asyncio.run(add_library(sessions, corpus, embedder))


def test_an_interview_follows_up_on_what_answers_missed_and_reports(
    client, sessions, library
) -> None:
    first_q, second_q, third_q = library

    started = client.post("/interviews", json={"size": 3})

    assert started.status_code == 201
    interview = started.json()
    assert (interview["status"], interview["size"], interview["report"]) == ("asking", 3, None)
    [first] = interview["turns"]
    assert (first["kind"], first["question_id"], first["topic"]) == (
        "question",
        first_q,
        "attention",
    )
    assert (interview["waiting"], first["time_limit"], first["attempt"]) == (first["id"], 180, None)
    # Its place is kept in the database between requests, not in the process.
    assert count(sessions, CHECKPOINTS) > 0

    # Half right: the follow-up asks about the key point the answer missed.
    interview = answer(client, interview, "It is GAP").json()
    first, follow = interview["turns"]
    assert first["attempt"]["grades"][0]["score"] == 0.6667
    assert first["no_follow_up"] is None
    assert follow["kind"] == "follow_up"
    assert follow["aim"] == {"kind": "missing", "text": "it keeps gradients large"}
    assert follow["text"] == "What did your answer leave out that the explanation depends on?"
    assert interview["waiting"] == follow["id"]
    # The graph was resumed with ids alone: the answer is nowhere in its saved place.
    assert count(sessions, CHECKPOINTS) > 0
    assert not kept_in_places(sessions, "It is GAP")

    interview = answer(client, interview, "FULL: it keeps them large").json()
    follow = interview["turns"][1]
    assert (follow["answer"], follow["grade"]["score"]) == ("FULL: it keeps them large", 1.0)
    assert follow["grade"]["key_points"][0]["text"] == SCALING
    second = interview["turns"][2]
    assert (second["question_id"], interview["waiting"]) == (second_q, second["id"])

    # Everything covered: nothing to follow up.
    interview = answer(client, interview, "FULL").json()
    assert interview["turns"][2]["no_follow_up"] == routing.COVERED
    third = interview["turns"][3]
    assert (third["question_id"], interview["waiting"]) == (third_q, third["id"])

    # A claim the passage contradicts comes first.
    interview = answer(client, interview, "WRONG").json()
    follow = interview["turns"][4]
    assert follow["aim"]["kind"] == "contradicted"
    assert follow["aim"]["text"] == "It works the other way round."

    interview = answer(client, interview, "MISS").json()

    assert (interview["status"], interview["waiting"]) == ("finished", None)
    found = interview["report"]
    scores = [
        turn["attempt"]["grades"][0]["score"] for turn in interview["turns"] if turn["attempt"]
    ]
    assert scores == [0.6667, 1.0, 0.5167]
    assert (found["answered"], found["mean_score"]) == (3, round(sum(scores) / 3, 4))
    assert (found["follow_ups"], found["recovered"]) == (2, 1)
    assert [topic["name"] for topic in found["review"]] == ["positions"]
    assert [round_["recovered"] for round_ in found["rounds"]] == [True, None, False]
    assert all(round_["rating"] and round_["due"] for round_ in found["rounds"])
    # Answers to library questions are practice; follow-ups count toward the limits alone.
    assert (count(sessions, Attempt), count(sessions, Review)) == (3, 3)
    assert count(sessions, GradeRequest) == 5
    # Each slot, follow-ups' included, records the model that replied, for the pacer.
    assert asyncio.run(charged(sessions)) == [("test-grader", 1)] * 5
    assert client.get("/practice/progress").json()["answers"] == 3
    # Over, it keeps no place.
    assert count(sessions, CHECKPOINTS) == 0
    assert client.get(f"/interviews/{interview['id']}").json() == interview
    [listed] = client.get("/interviews").json()
    assert (listed["id"], listed["status"], listed["answered"]) == (interview["id"], "finished", 3)


def test_questions_come_from_one_topic_or_across_topics(
    client, sessions, library, embedder
) -> None:
    extra = asyncio.run(add_library_question(sessions, library[0]))
    topic = asyncio.run(topic_of(sessions, library[0]))

    across = client.post("/interviews", json={"size": 3}).json()
    one = client.post("/interviews", json={"size": 3, "topic_id": topic}).json()

    assert asyncio.run(planned(sessions, across["id"])) == library
    assert asyncio.run(planned(sessions, one["id"])) == [library[0], extra]
    assert one["topic"] == {"id": topic, "name": "attention"}
    assert client.post("/interviews", json={"size": 3, "topic_id": 999}).status_code == 404
    assert client.post("/interviews", json={"size": 4}).status_code == 422


async def add_library_question(sessions, like: int) -> int:
    async with sessions() as session, session.begin():
        original = await session.get_one(Question, like)
        question = Question(
            text="Another question on the same passage?",
            reference_answer=original.reference_answer,
            key_points=original.key_points,
            style="why_how",
            difficulty=3,
            topic_id=original.topic_id,
            generator_model="openai/gpt-oss-120b",
            prompt_version="generate-v8",
        )
        session.add(question)
        await session.flush()
        chunk = original.key_points[0]["chunk_id"]
        session.add(QuestionSource(question_id=question.id, chunk_id=chunk, position=0))
        return question.id


async def topic_of(sessions, question_id: int) -> int:
    async with sessions() as session:
        return (await session.get_one(Question, question_id)).topic_id


async def planned(sessions, interview_id: int) -> list[int]:
    async with sessions() as session:
        return (await session.get_one(Interview, interview_id)).questions


def standing(question_id: int, topic_id: int | None, due: date | None = None) -> Standing:
    practised = due is not None
    return Standing(
        question_id=question_id,
        topic_id=topic_id,
        difficulty=2,
        score=0.5 if practised else None,
        state={} if practised else None,
        due=due,
        retrievability=0.5 if practised else 0.0,
    )


def test_an_interview_asks_what_is_due_first_then_spreads_across_topics() -> None:
    day = date(2026, 10, 9)
    questions = [
        standing(1, 10),
        standing(2, 10, due=day - timedelta(days=1)),
        standing(3, 20),
        standing(4, 30),
        standing(5, 10),
    ]

    assert plan(questions, day, 3, None) == [2, 3, 4]
    assert plan(questions, day, 5, None) == [2, 3, 4, 1, 5]
    assert plan(questions, day, 3, 10) == [2, 1, 5]
    assert plan(questions, day, 3, 30) == [4]
    assert plan([], day, 3, None) == []


def grade(score: float | None, labels: list[str], contradicted: bool = False) -> Grade:
    return Grade(
        status="graded" if score is not None else "failed",
        score=score,
        key_points=[{"id": f"k{n}", "status": s} for n, s in enumerate(labels, start=1)],
        claims=[{"claim": "No.", "verdict": "contradicted", "why": "Yes.", "chunk_id": 7}]
        if contradicted
        else [],
    )


POINTS = [
    {"text": "light", "weight": 1, "evidence_quote": "a", "chunk_id": 7},
    {"text": "heavy", "weight": 2, "evidence_quote": "b", "chunk_id": 7},
    {"text": "also heavy", "weight": 2, "evidence_quote": "c", "chunk_id": 7},
]


def test_a_follow_up_aims_at_an_error_then_at_the_weightiest_point_missed() -> None:
    contradicted = routing.aim(grade(0.9, ["covered"] * 3, contradicted=True), POINTS)
    assert isinstance(contradicted, routing.Gap) and contradicted.kind == "contradicted"
    assert contradicted.chunk_id == 7

    heaviest = routing.aim(grade(0.2, ["missing", "partial", "missing"]), POINTS)
    assert isinstance(heaviest, routing.Gap)
    assert (heaviest.kind, heaviest.text, heaviest.point["id"]) == ("missing", "also heavy", "k3")
    first_of_equals = routing.aim(grade(0.2, ["covered", "missing", "missing"]), POINTS)
    assert isinstance(first_of_equals, routing.Gap) and first_of_equals.text == "heavy"

    assert routing.aim(grade(0.85, ["covered", "covered", "partial"]), POINTS) == routing.COVERED
    assert routing.aim(grade(None, []), POINTS) == routing.NOT_GRADED
    assert routing.aim(None, POINTS) == routing.NOT_GRADED


def test_the_report_names_the_topics_a_score_or_a_follow_up_fell_short_in() -> None:
    def round_(topic: int, score: float | None, follow_up: float | None = None) -> Round:
        return Round(1, topic, True, score, follow_up, "good", None)

    found = report(
        [round_(1, 0.9), round_(2, 0.5, 0.8), round_(3, 0.7, 0.4), round_(2, 0.3), round_(4, None)]
    )

    assert (found.answered, found.follow_ups, found.recovered) == (5, 2, 1)
    assert found.mean_score == round((0.9 + 0.5 + 0.7 + 0.3) / 4, 4)
    assert found.review == [2, 3]


def test_an_answer_past_a_limit_is_refused_and_nothing_is_written(
    client, sessions, library, settings
) -> None:
    limited(settings, per_user=1)
    interview = client.post("/interviews", json={"size": 3}).json()

    # The one grade goes to the answer, so no follow-up is written for it.
    interview = answer(client, interview, "It is GAP").json()
    first, second = interview["turns"]
    assert first["no_follow_up"] == routing.NO_GRADE_LEFT
    assert interview["waiting"] == second["id"]

    refused = answer(client, interview, "FULL")

    assert refused.status_code == 429
    assert refused.json()["limit"]["scope"] == "per_user"
    assert "Retry-After" in refused.headers
    again = client.get(f"/interviews/{interview['id']}").json()
    assert again == interview
    assert count(sessions, Attempt) == 1
    # Nor does an interview start while no grade is left.
    assert client.post("/interviews", json={"size": 3}).status_code == 429


def test_without_a_writer_an_interview_goes_on_without_follow_ups(client, library) -> None:
    use_writer(None)
    interview = client.post("/interviews", json={"size": 3}).json()

    interview = answer(client, interview, "It is GAP").json()

    first, second = interview["turns"]
    assert (first["no_follow_up"], second["kind"]) == (routing.NO_WRITER, "question")


def test_a_follow_up_that_fails_its_checks_is_left_out(client, sessions, library) -> None:
    async def failing(question, sources, gap, answer):
        return Writing(
            None, "test-writer", usage={"requests": 2, "input_tokens": 900, "output_tokens": 300}
        )

    use_writer(failing)
    interview = client.post("/interviews", json={"size": 3}).json()

    interview = answer(client, interview, "It is GAP").json()

    assert interview["turns"][0]["no_follow_up"] == routing.NOT_WRITTEN
    # What trying cost still counts toward the writer's day.
    assert asyncio.run(spent(sessions))["test-writer"] == (2, 1200)


def test_an_answer_no_model_could_grade_is_kept_and_not_followed_up(
    client, sessions, library
) -> None:
    interview = client.post("/interviews", json={"size": 3}).json()

    interview = answer(client, interview, "FAIL").json()

    first, second = interview["turns"]
    assert first["attempt"]["grades"][0]["status"] == "failed"
    assert first["no_follow_up"] == routing.NOT_GRADED
    assert interview["waiting"] == second["id"]
    # No model replied, so the grade cost nothing and its slot was given back.
    assert count(sessions, GradeRequest) == 0


def test_only_the_question_waiting_can_be_answered_and_not_once_it_is_over(client, library) -> None:
    interview = client.post("/interviews", json={"size": 3}).json()
    stale = dict(interview, waiting=interview["waiting"] + 100)

    assert answer(client, stale, "FULL").status_code == 409

    answered = answer(client, interview, "FULL").json()
    # The same question again, as a second click would send it
    assert answer(client, interview, "FULL").status_code == 409

    ended = client.post(f"/interviews/{interview['id']}/end").json()
    assert (ended["status"], ended["waiting"]) == ("ended", None)
    assert ended["report"]["answered"] == 1
    assert answer(client, answered, "FULL").status_code == 409


def test_one_answer_sent_twice_at_once_is_taken_once(
    client, sessions, library, monkeypatch
) -> None:
    interview = client.post("/interviews", json={"size": 3}).json()
    saving = interviews_api.save_answer

    async def slowly(*args, **kwargs):
        # Long enough for the other request to arrive while this one is saving
        await asyncio.sleep(0.3)
        return await saving(*args, **kwargs)

    monkeypatch.setattr(interviews_api, "save_answer", slowly)
    body = {"turn_id": interview["waiting"], "answer": "FULL", "seconds": 5.0}

    async def twice() -> list[int]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as browser:
            sent = [browser.post(f"/interviews/{interview['id']}/answers", json=body)]
            sent.append(browser.post(f"/interviews/{interview['id']}/answers", json=body))
            return sorted(response.status_code for response in await asyncio.gather(*sent))

    assert asyncio.run(twice()) == [200, 409]
    assert count(sessions, Attempt) == 1


def test_an_interview_is_its_users_own(client, sessions, library) -> None:
    someone = asyncio.run(add_user(sessions, "someone"))

    async def theirs() -> int:
        async with sessions() as session, session.begin():
            interview = Interview(user_id=someone, questions=library)
            session.add(interview)
            await session.flush()
            return interview.id

    other = asyncio.run(theirs())

    assert client.get(f"/interviews/{other}").status_code == 404
    assert client.post(f"/interviews/{other}/end").status_code == 404
    assert answer(client, {"id": other, "waiting": 1}, "FULL").status_code == 404
    assert client.get("/interviews").json() == []


def test_deleting_the_practice_deletes_interviews_and_their_places(
    client, sessions, library
) -> None:
    interview = client.post("/interviews", json={"size": 3}).json()
    answer(client, interview, "It is GAP")
    assert count(sessions, CHECKPOINTS) > 0

    deleted = client.delete("/practice").json()

    assert (deleted["interviews"], deleted["attempts"]) == (1, 1)
    assert count(sessions, Interview) == count(sessions, InterviewTurn) == 0
    assert count(sessions, CHECKPOINTS) == 0


def test_an_interview_a_failed_request_left_halfway_is_carried_on(
    client, sessions, library
) -> None:
    interview = client.post("/interviews", json={"size": 3}).json()
    use_grader(interviewer_grader(crash=True))

    with pytest.raises(RuntimeError):
        answer(client, interview, "FULL")

    # The answer is kept, ungraded, and nothing is marked as moving the interview on.
    async def halfway() -> tuple[int, int, datetime | None]:
        async with sessions() as session:
            found = await session.get_one(Interview, interview["id"])
            graded = await session.scalar(select(func.count()).select_from(Grade))
            return (
                await session.scalar(select(func.count()).select_from(Attempt)),
                graded,
                found.moving_since,
            )

    assert asyncio.run(halfway()) == (1, 0, None)
    use_grader(interviewer_grader())

    async def mark(since: datetime | None) -> None:
        async with sessions() as session, session.begin():
            await session.execute(update(Interview).values(moving_since=since))

    # While a request is still moving it on, the next look leaves it alone...
    asyncio.run(mark(datetime.now(UTC)))
    assert client.get(f"/interviews/{interview['id']}").json()["waiting"] is None
    assert client.post(f"/interviews/{interview['id']}/end").status_code == 409

    # ...and once that request must have failed, the next look carries it on.
    asyncio.run(mark(datetime.now(UTC) - timedelta(minutes=6)))
    carried = client.get(f"/interviews/{interview['id']}").json()

    first, second = carried["turns"]
    assert first["attempt"]["grades"][0]["score"] == 1.0
    assert carried["waiting"] == second["id"]


def test_langsmith_traces_nothing_whatever_the_environment_says(
    client, library, monkeypatch
) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-test")
    langsmith_utils.get_env_var.cache_clear()
    seen = []

    async def watching(question, sources, gap, answer):
        seen.append((langsmith_utils.tracing_is_enabled(), get_tracing_context()["enabled"]))
        return await FollowUpWriter(fakes.follow_up_writer())(question, sources, gap, answer)

    use_writer(watching)
    try:
        assert langsmith_utils.tracing_is_enabled() is True
        interview = client.post("/interviews", json={"size": 3}).json()
        answer(client, interview, "It is GAP")
    finally:
        monkeypatch.undo()
        langsmith_utils.get_env_var.cache_clear()

    assert seen == [(False, False)]


def test_the_saver_finds_its_tables_as_it_would_set_them_up(database_url) -> None:
    """Migration 0012 lays out LangGraph's tables itself; they have to be what the saver's own
    set-up makes, record of its migrations included, or a new version of the saver would find
    them wrong."""
    url = make_url(database_url).set(drivername="postgresql")
    admin = url.set(database="postgres").render_as_string(hide_password=False)
    ours = url.render_as_string(hide_password=False)
    theirs = url.set(database="daedalus_saver").render_as_string(hide_password=False)
    with psycopg.connect(admin, autocommit=True) as connection:
        connection.execute("DROP DATABASE IF EXISTS daedalus_saver WITH (FORCE)")
        connection.execute("CREATE DATABASE daedalus_saver")

    async def set_up(conninfo: str) -> None:
        async with await AsyncConnection.connect(
            conninfo, autocommit=True, row_factory=dict_row
        ) as connection:
            await AsyncPostgresSaver(connection).setup()

    def shape(conninfo: str):
        with psycopg.connect(conninfo) as connection:
            return (
                connection.execute(
                    "SELECT table_name, column_name, data_type, is_nullable, column_default "
                    "FROM information_schema.columns WHERE table_name LIKE 'checkpoint%' "
                    "ORDER BY 1, 2"
                ).fetchall(),
                connection.execute(
                    "SELECT tablename, indexname, indexdef FROM pg_indexes "
                    "WHERE tablename LIKE 'checkpoint%' ORDER BY 1, 2"
                ).fetchall(),
                connection.execute(
                    "SELECT array_agg(v ORDER BY v) FROM checkpoint_migrations"
                ).fetchone()[0],
            )

    try:
        asyncio.run(set_up(theirs))
        expected = shape(theirs)
        assert shape(ours) == expected
        # Set up again on ours, the saver finds nothing to do.
        asyncio.run(set_up(ours))
        assert shape(ours) == expected
    finally:
        with psycopg.connect(admin, autocommit=True) as connection:
            connection.execute("DROP DATABASE IF EXISTS daedalus_saver WITH (FORCE)")

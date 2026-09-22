"""The grading endpoints: answering a question, reading attempts back, grading again."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import SecretStr
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile

from app.api.grading import MAX_ANSWER_CHARS, get_grader
from app.core.config import Settings
from app.db.models import Attempt, Grade, Question, QuestionSource
from app.main import app

ANSWER = "The dot products grow with the key dimension, so they are scaled to keep gradients."
KEY_POINTS = [
    {
        "text": "large dot products shrink the softmax gradients",
        "weight": 2,
        "evidence_quote": "which keeps the softmax gradients large",
        "chunk_id": None,
    },
    {
        "text": "the scale is the square root of the key dimension",
        "weight": 1,
        "evidence_quote": "by the square root of the key dimension",
        "chunk_id": None,
    },
]


async def add_question(sessions, chunk_id: int, status: str = "accepted") -> int:
    async with sessions() as session, session.begin():
        question = Question(
            text="Why are the dot products divided by the square root of the key dimension?",
            reference_answer="Large values push the softmax into a region with tiny gradients.",
            key_points=KEY_POINTS,
            style="why_how",
            difficulty=3,
            status=status,
            generator_model="openai/gpt-oss-120b",
            prompt_version="generate-v3",
        )
        session.add(question)
        await session.flush()
        session.add(QuestionSource(question_id=question.id, chunk_id=chunk_id, position=0))
        return question.id


def grader_output(chunk_id: int) -> dict:
    return {
        "key_points": [
            {"id": "k1", "status": "covered", "answer_quote": "so they are scaled"},
            {"id": "k2", "status": "missing", "answer_quote": ""},
        ],
        "claims": [
            {
                "claim": "The dot products grow with the key dimension.",
                "verdict": "supported",
                "chunk_id": chunk_id,
                "why": "The passage divides them by the square root of the key dimension.",
            },
            {
                "claim": "Scaling is also cheaper to compute.",
                "verdict": "unverified",
                "chunk_id": 0,
                "why": "The passage says nothing about cost.",
            },
        ],
        "clarity": 4,
        "strengths": ["names the growth of the dot products"],
        "gaps": ["does not say what the scale is"],
        "errors": [],
        "improved_answer": "Dividing by the square root of the key dimension keeps gradients.",
        "follow_up": "What would happen to the softmax without the scaling?",
    }


def grading(chunk_id: int) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(json.dumps(grader_output(chunk_id)))])

    return FunctionModel(respond, profile=ModelProfile(supports_json_schema_output=True))


def refusing() -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status_code=429, model_name="grader", body="rate limit reached")

    return FunctionModel(respond, profile=ModelProfile(supports_json_schema_output=True))


def use_grader(model: FunctionModel) -> None:
    # The client fixture clears every override when the test ends.
    app.dependency_overrides[get_grader] = lambda: model


def test_an_answer_comes_back_graded_with_its_claims_cited(client, sessions, corpus) -> None:
    question_id = asyncio.run(add_question(sessions, corpus.scaling))
    use_grader(grading(corpus.scaling))

    response = client.post(f"/questions/{question_id}/attempts", json={"answer": ANSWER})

    assert response.status_code == 201
    attempt = response.json()
    assert (attempt["question_id"], attempt["answer"]) == (question_id, ANSWER)
    [grade] = attempt["grades"]
    assert grade["status"] == "graded"
    # Weight 2 covered, weight 1 missing, nothing contradicted
    assert (grade["coverage"], grade["contradicted"], grade["score"]) == (0.6667, 0, 0.6667)
    assert [(point["text"], point["weight"], point["status"]) for point in grade["key_points"]] == [
        (KEY_POINTS[0]["text"], 2, "covered"),
        (KEY_POINTS[1]["text"], 1, "missing"),
    ]
    assert grade["key_points"][0]["quote_found"] is True
    supported, unverified = grade["claims"]
    assert supported["chunk_id"] == corpus.scaling
    assert supported["citation"].startswith("Attention Is All You Need")
    assert supported["link"] == "https://arxiv.org/html/1706.03762v7#S3.SS2.SSS1"
    # An unverified claim has no passage behind it.
    assert [unverified[field] for field in ("chunk_id", "citation", "link")] == [None] * 3
    assert grade["follow_up"] and grade["improved_answer"]


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param("", id="empty"),
        pytest.param("   \n ", id="blank"),
        pytest.param("x" * (MAX_ANSWER_CHARS + 1), id="too long"),
    ],
)
def test_an_answer_has_to_say_something_and_not_too_much(client, sessions, corpus, answer) -> None:
    question_id = asyncio.run(add_question(sessions, corpus.scaling))
    use_grader(grading(corpus.scaling))

    response = client.post(f"/questions/{question_id}/attempts", json={"answer": answer})

    assert response.status_code == 422


def test_an_answer_keeps_how_long_it_took(client, sessions, corpus) -> None:
    question_id = asyncio.run(add_question(sessions, corpus.scaling))
    use_grader(grading(corpus.scaling))

    timed = {"answer": ANSWER, "seconds": 95.5, "time_limit": 180}
    attempt = client.post(f"/questions/{question_id}/attempts", json=timed).json()
    untimed = client.post(f"/questions/{question_id}/attempts", json={"answer": ANSWER}).json()

    assert (attempt["seconds"], attempt["time_limit"]) == (95.5, 180)
    assert client.get(f"/attempts/{attempt['id']}").json()["seconds"] == 95.5
    assert (untimed["seconds"], untimed["time_limit"]) == (None, None)


@pytest.mark.parametrize(
    "timing",
    [
        pytest.param({"seconds": -1}, id="negative time"),
        pytest.param({"seconds": 2 * 24 * 60 * 60}, id="two days"),
        pytest.param({"time_limit": 0}, id="no time at all"),
    ],
)
def test_an_answer_s_timing_has_to_make_sense(client, sessions, corpus, timing) -> None:
    question_id = asyncio.run(add_question(sessions, corpus.scaling))
    use_grader(grading(corpus.scaling))

    response = client.post(f"/questions/{question_id}/attempts", json={"answer": ANSWER} | timing)

    assert response.status_code == 422


@pytest.mark.parametrize("state", ["rejected", "retired", None])
def test_only_a_question_in_the_library_can_be_answered(client, sessions, corpus, state) -> None:
    question_id = asyncio.run(add_question(sessions, corpus.scaling, state)) if state else 999
    use_grader(grading(corpus.scaling))

    response = client.post(f"/questions/{question_id}/attempts", json={"answer": ANSWER})

    assert response.status_code == 404


def test_an_answer_no_model_could_grade_is_kept_and_says_why(client, sessions, corpus) -> None:
    question_id = asyncio.run(add_question(sessions, corpus.scaling))
    use_grader(refusing())

    response = client.post(f"/questions/{question_id}/attempts", json={"answer": ANSWER})

    assert response.status_code == 201
    [grade] = response.json()["grades"]
    assert (grade["status"], grade["score"]) == ("failed", None)
    assert "status_code: 429" in grade["error"]
    stored = client.get(f"/attempts/{response.json()['id']}").json()
    assert stored["answer"] == ANSWER and stored["grades"][0]["status"] == "failed"


def test_a_failed_attempt_can_be_graded_again(client, sessions, corpus) -> None:
    question_id = asyncio.run(add_question(sessions, corpus.scaling))
    use_grader(refusing())
    failed = client.post(f"/questions/{question_id}/attempts", json={"answer": ANSWER})
    attempt_id = failed.json()["id"]
    use_grader(grading(corpus.scaling))

    response = client.post(f"/attempts/{attempt_id}/grades")

    assert response.status_code == 201
    assert [grade["status"] for grade in response.json()["grades"]] == ["failed", "graded"]
    assert client.post("/attempts/999/grades").status_code == 404


def test_attempts_are_listed_newest_first(client, sessions, corpus) -> None:
    question_id = asyncio.run(add_question(sessions, corpus.scaling))
    use_grader(grading(corpus.scaling))
    first = client.post(f"/questions/{question_id}/attempts", json={"answer": "first"}).json()
    second = client.post(f"/questions/{question_id}/attempts", json={"answer": "second"}).json()

    listed = client.get(f"/questions/{question_id}/attempts").json()

    assert [attempt["id"] for attempt in listed] == [second["id"], first["id"]]
    assert client.get(f"/questions/{question_id}/attempts?limit=1").json()[0]["id"] == second["id"]
    assert client.get("/questions/999/attempts").status_code == 404
    assert client.get("/attempts/999").status_code == 404


def test_the_grader_is_built_once_from_what_the_day_has_spent(sessions, corpus) -> None:
    """Gemini writes questions and grades answers out of one allowance, so both count."""
    settings = Settings(_env_file=None, groq_api_key=SecretStr("k"), gemini_api_key=SecretStr("k"))

    async def scenario():
        question_id = await add_question(sessions, corpus.scaling)
        async with sessions() as session, session.begin():
            question = await session.get_one(Question, question_id)
            question.generator_model = "gemini-3.5-flash"
            question.usage = {"requests": 1, "input_tokens": 1500, "output_tokens": 800}
            attempt = Attempt(question_id=question_id, answer=ANSWER)
            session.add(attempt)
            await session.flush()
            for model, output in (("qwen/qwen3.8-27b", 600), ("gemini-3.5-flash", 700)):
                session.add(
                    Grade(
                        attempt_id=attempt.id,
                        status="graded",
                        grader_model=model,
                        prompt_version="grade-v1",
                        clarity=4,
                        coverage=1.0,
                        contradicted=0,
                        score=1.0,
                        usage={"requests": 1, "input_tokens": 1100, "output_tokens": output},
                    )
                )
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
        async with sessions() as session:
            first = await get_grader(request, session, settings)
            second = await get_grader(request, session, settings)
        return first, second

    first, second = asyncio.run(scenario())

    assert first is second
    qwen, _, gemini = first.models
    assert qwen.pacer.spent == (1, 1_700)
    assert gemini.pacer.spent == (2, 2_300 + 1_800)

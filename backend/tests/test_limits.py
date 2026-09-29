"""Daily limits on grading: so many grades a user a practice day, so many in all over the last
24 hours, a refusal that spends nothing, and a count that survives deleting practice."""

import asyncio
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile
from sqlalchemy import delete, func, select
from test_grading_api import ANSWER, add_question, grading, refusing, use_grader
from test_users import act_as, add_user, builtin_user

from app.core.config import Settings
from app.db.models import Attempt, Grade, GradeRequest
from app.grading.grader import spent_today
from app.grading.limits import KEPT, allowance, day_bounds, take
from app.main import app

# A Tuesday morning, well inside its practice day wherever it is looked at from
NOW = datetime(2026, 9, 29, 10, 0, tzinfo=UTC)


def limited(settings: Settings, per_user: int | None = None, in_all: int | None = None) -> None:
    settings.daily_grades_per_user = per_user
    settings.daily_grades = in_all


def counted(model: FunctionModel) -> tuple[FunctionModel, list[int]]:
    """The model, and a list that grows by one each time it is asked."""
    calls: list[int] = []
    function = model.function
    assert function is not None

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(1)
        return function(messages, info)  # type: ignore[return-value]

    return FunctionModel(respond, profile=model.profile), calls


def unreadable() -> FunctionModel:
    """A grader that replies, every time, with something that isn't a grade."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("not a grade")])

    return FunctionModel(
        respond, model_name="unreadable", profile=ModelProfile(supports_json_schema_output=True)
    )


async def count(sessions, table) -> int:
    async with sessions() as session:
        return await session.scalar(select(func.count()).select_from(table)) or 0


async def add_requests(sessions, user_id: int | None, *moments: datetime) -> None:
    async with sessions() as session, session.begin():
        session.add_all(
            GradeRequest(user_id=user_id, model="m", requests=1, created_at=moment)
            for moment in moments
        )


def test_a_user_has_so_many_grades_a_practice_day(client, sessions, corpus, settings) -> None:
    limited(settings, per_user=2)
    question_id = asyncio.run(add_question(sessions, corpus.scaling))
    model, calls = counted(grading(corpus.scaling))
    use_grader(model)

    answers = [
        client.post(f"/questions/{question_id}/attempts", json={"answer": ANSWER}) for _ in range(3)
    ]

    assert [answer.status_code for answer in answers] == [201, 201, 429]
    refused = answers[-1]
    _, next_start = day_bounds(datetime.now(UTC), settings)
    assert refused.json() == {
        "detail": "You have used today's 2 grades; more from 04:00 UTC.",
        "limit": {
            "scope": "per_user",
            "allowed": 2,
            "used": 2,
            "left": 0,
            "again_at": next_start.isoformat().replace("+00:00", "Z"),
        },
    }
    wait = (next_start - datetime.now(UTC)).total_seconds()
    assert abs(int(refused.headers["retry-after"]) - wait) < 5
    # Refused before anything was written or asked: the answer stays a draft in the browser.
    assert len(calls) == 2
    assert asyncio.run(count(sessions, Attempt)) == 2
    assert asyncio.run(count(sessions, GradeRequest)) == 2


def test_grading_again_counts_like_any_grade(client, sessions, corpus, settings) -> None:
    limited(settings, per_user=2)
    question_id = asyncio.run(add_question(sessions, corpus.scaling))
    use_grader(grading(corpus.scaling))
    attempt = client.post(f"/questions/{question_id}/attempts", json={"answer": ANSWER}).json()

    again = [client.post(f"/attempts/{attempt['id']}/grades") for _ in range(2)]

    assert [response.status_code for response in again] == [201, 429]
    assert len(client.get(f"/attempts/{attempt['id']}").json()["grades"]) == 2


def test_everyone_together_has_so_many_over_a_day(client, sessions, corpus, settings) -> None:
    limited(settings, per_user=5, in_all=2)
    question_id = asyncio.run(add_question(sessions, corpus.scaling))
    use_grader(grading(corpus.scaling))
    first = client.post(f"/questions/{question_id}/attempts", json={"answer": ANSWER})
    act_as(asyncio.run(add_user(sessions, "someone")))
    second = client.post(f"/questions/{question_id}/attempts", json={"answer": ANSWER})
    act_as(asyncio.run(add_user(sessions, "someone else")))

    third = client.post(f"/questions/{question_id}/attempts", json={"answer": ANSWER})

    assert (first.status_code, second.status_code, third.status_code) == (201, 201, 429)
    limit = third.json()["limit"]
    assert (limit["scope"], limit["allowed"], limit["used"], limit["left"]) == ("in_all", 2, 2, 0)
    # The next grade is free once the first leaves the last 24 hours.
    made = datetime.fromisoformat(first.json()["grades"][0]["created_at"])
    again = datetime.fromisoformat(limit["again_at"])
    assert abs((again - made - timedelta(days=1)).total_seconds()) < 5
    assert third.json()["detail"].startswith("All 2 grades the last 24 hours allow have been used")
    # Their own day is untouched: the others' grades were the others'.
    assert client.get("/practice/allowance").json()["per_user"]["used"] == 0


def test_a_user_s_count_starts_again_with_the_practice_day(sessions) -> None:
    """A practice day starts at 04:00 in the practice time zone: grades before that were
    yesterday's."""
    settings = Settings(_env_file=None, practice_timezone="Asia/Kolkata", daily_grades_per_user=2)
    start, next_start = day_bounds(NOW, settings)
    user_id = builtin_user(sessions)

    async def scenario():
        await add_requests(sessions, user_id, start - timedelta(seconds=1), start)
        async with sessions() as session:
            before = await allowance(session, settings, user_id, NOW)
        await add_requests(sessions, user_id, NOW - timedelta(minutes=1))
        async with sessions() as session:
            after = await allowance(session, settings, user_id, NOW)
            tomorrow = await allowance(session, settings, user_id, next_start)
        return before.per_user, after.per_user, tomorrow.per_user

    before, after, tomorrow = asyncio.run(scenario())

    # 10:00 UTC is 15:30 in India, so the day started at 04:00 there, 22:30 UTC the day before.
    assert (start, next_start) == (
        datetime(2026, 9, 29, 4, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
        datetime(2026, 9, 30, 4, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
    )
    assert (before.used, before.left, before.again_at) == (1, 1, None)
    assert (after.used, after.left, after.again_at) == (2, 0, next_start)
    assert (tomorrow.used, tomorrow.left) == (0, 2)


def test_the_count_in_all_looks_back_24_hours(sessions) -> None:
    settings = Settings(_env_file=None, daily_grades=2)
    user_id = builtin_user(sessions)
    oldest, older, newest = (NOW - timedelta(hours=hours) for hours in (23, 20, 1))

    async def scenario():
        await add_requests(sessions, user_id, NOW - timedelta(days=1), oldest, older)
        async with sessions() as session:
            full = await allowance(session, settings, user_id, NOW)
        await add_requests(sessions, None, newest)
        async with sessions() as session:
            over = await allowance(session, settings, user_id, NOW)
        return full.in_all, over.in_all

    full, over = asyncio.run(scenario())

    # The grade exactly a day old has left; one more is free when the oldest counted does.
    assert (full.used, full.left, full.again_at) == (2, 0, oldest + timedelta(days=1))
    # Past the limit (it was lowered, say), enough of the oldest have to leave first.
    assert (over.used, over.left, over.again_at) == (3, 0, older + timedelta(days=1))


@pytest.mark.parametrize(
    "in_all, before_the_day, refused_by",
    [
        # Both used up: the one that frees a grade later refuses it.
        pytest.param(1, False, "in_all", id="everyone's frees later"),
        pytest.param(2, True, "per_user", id="yours frees later"),
    ],
)
def test_a_grade_is_refused_by_the_limit_that_allows_one_again_later(
    sessions, in_all, before_the_day, refused_by
) -> None:
    settings = Settings(_env_file=None, daily_grades_per_user=1, daily_grades=in_all)
    start, _ = day_bounds(NOW, settings)
    user_id = builtin_user(sessions)
    moments = [start + timedelta(hours=1)]
    if before_the_day:
        moments.insert(0, start - timedelta(hours=1))

    async def scenario():
        await add_requests(sessions, user_id, *moments)
        async with sessions() as session:
            return (await allowance(session, settings, user_id, NOW)).refusal()

    refusal = asyncio.run(scenario())

    assert refusal is not None and refusal.scope == refused_by


def test_a_limit_of_nought_switches_grading_off(client, sessions, corpus, settings) -> None:
    limited(settings, in_all=0)
    question_id = asyncio.run(add_question(sessions, corpus.scaling))
    use_grader(grading(corpus.scaling))

    refused = client.post(f"/questions/{question_id}/attempts", json={"answer": ANSWER})

    assert refused.status_code == 429
    assert refused.json()["detail"] == "Grading is switched off here for now."
    assert refused.json()["limit"]["again_at"] is None
    assert "retry-after" not in refused.headers


def test_a_failed_grade_counts_only_when_a_model_replied(
    client, sessions, corpus, settings
) -> None:
    limited(settings, per_user=5)
    question_id = asyncio.run(add_question(sessions, corpus.scaling))
    use_grader(refusing())
    unanswered = client.post(f"/questions/{question_id}/attempts", json={"answer": ANSWER}).json()
    use_grader(unreadable())

    answered = client.post(f"/questions/{question_id}/attempts", json={"answer": ANSWER}).json()

    assert unanswered["grades"][0]["usage"] == {}
    # The reply and the one retry it was allowed both cost tokens.
    [grade] = answered["grades"]
    assert grade["status"] == "failed" and grade["grader_model"] is None
    assert grade["usage"]["requests"] == 2 and grade["usage"]["input_tokens"] > 0
    assert client.get("/practice/allowance").json()["per_user"]["used"] == 1

    async def charged():
        async with sessions() as session:
            [row] = await session.scalars(select(GradeRequest))
            return row, await spent_today(session)

    row, spent = asyncio.run(charged())
    assert (row.grade_id, row.model, row.requests) == (grade["id"], "unreadable", 2)
    # The pacer starts from it too.
    assert spent["unreadable"] == (2, row.input_tokens + row.output_tokens)


def test_deleting_practice_leaves_the_day_s_grades_counted(
    client, sessions, corpus, settings
) -> None:
    limited(settings, per_user=2)
    question_id = asyncio.run(add_question(sessions, corpus.scaling))
    use_grader(grading(corpus.scaling))
    for _ in range(2):
        client.post(f"/questions/{question_id}/attempts", json={"answer": ANSWER})

    async def delete_practice():
        async with sessions() as session, session.begin():
            await session.execute(delete(Attempt))

    asyncio.run(delete_practice())

    assert asyncio.run(count(sessions, Grade)) == 0
    assert client.get("/practice/allowance").json()["per_user"]["used"] == 2
    refused = client.post(f"/questions/{question_id}/attempts", json={"answer": ANSWER})
    assert refused.status_code == 429


def test_rows_are_kept_only_as_long_as_the_limits_look_back(sessions, settings) -> None:
    user_id = builtin_user(sessions)

    async def scenario():
        await add_requests(sessions, user_id, NOW - KEPT - timedelta(seconds=1), NOW - KEPT)
        async with sessions() as session, session.begin():
            await take(session, settings, user_id, NOW)
        async with sessions() as session:
            return list(await session.scalars(select(GradeRequest.created_at)))

    assert sorted(asyncio.run(scenario())) == [NOW - KEPT, NOW]


def test_requests_sent_together_cannot_pass_on_one_count(
    client, sessions, corpus, settings
) -> None:
    """Grading takes seconds: without a slot taken under a lock first, every request sent at
    once would see the count before any grade was written."""
    limited(settings, per_user=1)
    question_id = asyncio.run(add_question(sessions, corpus.scaling))
    output = grading(corpus.scaling).function
    assert output is not None

    async def slow(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        await asyncio.sleep(0.3)
        return output(messages, info)  # type: ignore[return-value]

    use_grader(FunctionModel(slow, profile=ModelProfile(supports_json_schema_output=True)))

    async def together() -> list[int]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as api:
            responses = await asyncio.gather(
                *(
                    api.post(f"/questions/{question_id}/attempts", json={"answer": ANSWER})
                    for _ in range(4)
                )
            )
        return sorted(response.status_code for response in responses)

    assert asyncio.run(together()) == [201, 429, 429, 429]
    assert asyncio.run(count(sessions, Attempt)) == 1


def test_the_allowance_says_what_is_left(client, sessions, corpus, settings) -> None:
    question_id = asyncio.run(add_question(sessions, corpus.scaling))
    use_grader(grading(corpus.scaling))

    unlimited = client.get("/practice/allowance").json()
    limited(settings, per_user=2, in_all=5)
    client.post(f"/questions/{question_id}/attempts", json={"answer": ANSWER})
    some_left = client.get("/practice/allowance").json()
    client.post(f"/questions/{question_id}/attempts", json={"answer": ANSWER})
    none_left = client.get("/practice/allowance").json()

    assert unlimited == {"per_user": None, "in_all": None, "left": None, "refused_by": None}
    assert (some_left["per_user"]["left"], some_left["in_all"]["left"]) == (1, 4)
    assert (some_left["left"], some_left["refused_by"]) == (1, None)
    assert some_left["per_user"]["again_at"] is None
    assert none_left["left"] == 0
    assert none_left["refused_by"] == none_left["per_user"]
    assert none_left["per_user"]["again_at"] is not None


def test_the_demo_is_limited_and_a_local_app_is_not() -> None:
    local = Settings(_env_file=None)
    demo = Settings(_env_file=None, environment="production")
    switched_off = Settings(_env_file=None, environment="production", daily_grades=0)

    assert (local.daily_grades_per_user, local.daily_grades) == (None, None)
    assert (demo.daily_grades_per_user, demo.daily_grades) == (10, 80)
    assert (switched_off.daily_grades_per_user, switched_off.daily_grades) == (10, 0)


def test_both_grading_routes_document_the_refusal() -> None:
    paths = app.openapi()["paths"]

    for path in ("/questions/{question_id}/attempts", "/attempts/{attempt_id}/grades"):
        refusal = paths[path]["post"]["responses"]["429"]
        assert refusal["content"]["application/json"]["schema"]["$ref"].endswith("/RefusalOut")

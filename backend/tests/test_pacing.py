"""Holding a provider to its rates, and waiting out its refusals instead of abandoning it."""

import asyncio

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.llm.pacing import (
    Limits,
    PacedModel,
    Pacer,
    QuotaExhausted,
    asked_delay,
    estimate_tokens,
)

LIMITS = Limits(
    requests_per_minute=2, tokens_per_minute=1_000, requests_per_day=5, tokens_per_day=4_000
)


class FakeTime:
    """A clock that only moves when something sleeps, so the tests take no time at all."""

    def __init__(self) -> None:
        self.now = 1_000.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(round(seconds, 3))
        self.now += seconds


def a_pacer(limits: Limits = LIMITS) -> tuple[Pacer, FakeTime]:
    time = FakeTime()
    return Pacer("groq", limits, clock=time.clock, sleep=time.sleep), time


def refusal(status: int = 429, retry_after: str | None = "9", body: str = "") -> ModelHTTPError:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    return ModelHTTPError(status_code=status, model_name="groq", body=body, headers=headers)


def test_requests_wait_for_room_in_the_minute() -> None:
    pacer, time = a_pacer()

    async def scenario():
        for _ in range(3):
            await pacer.acquire(10)

    asyncio.run(scenario())

    # Two fit; the third waits for the first to fall out of the window.
    assert time.slept == [60.0]


def test_a_big_request_waits_for_the_token_ceiling() -> None:
    pacer, time = a_pacer()

    async def scenario():
        await pacer.acquire(600)
        await pacer.acquire(600)

    asyncio.run(scenario())

    assert time.slept == [60.0]


def test_the_estimate_is_corrected_by_what_was_really_spent() -> None:
    pacer, _ = a_pacer()

    async def scenario():
        entry = await pacer.acquire(900)
        pacer.record(entry, 320)
        return pacer.spent

    assert asyncio.run(scenario()) == (1, 320)


def test_running_out_for_the_day_reads_as_a_model_error() -> None:
    pacer, _ = a_pacer(Limits(30, 8_000, requests_per_day=1, tokens_per_day=200_000))

    async def scenario():
        await pacer.acquire(10)
        await pacer.acquire(10)

    with pytest.raises(QuotaExhausted, match="out of requests"):
        asyncio.run(scenario())


def test_a_refusal_is_waited_out_with_a_margin_on_top() -> None:
    pacer, time = a_pacer()

    asyncio.run(pacer.wait_out(refusal(retry_after="9"), 1))

    # Twice in the trial the provider's own figure was not long enough.
    assert time.slept == [11.0]


@pytest.mark.parametrize(
    ("exc", "delay"),
    [
        (refusal(retry_after="9"), 9.0),
        (refusal(retry_after=None, body="Limit 8000, Used 7900. Please try again in 9.6s"), 9.6),
        (refusal(retry_after=None, body="rate limit reached"), None),
        (refusal(retry_after="Wed, 21 Oct 2026 07:28:00 GMT"), None),
    ],
)
def test_the_delay_is_read_from_the_header_or_the_message(exc, delay) -> None:
    assert asked_delay(exc) == delay


def test_a_refusal_that_names_no_delay_backs_off_further_each_time() -> None:
    pacer, time = a_pacer()

    async def scenario():
        await pacer.wait_out(refusal(status=503, retry_after=None), 1)
        await pacer.wait_out(refusal(status=503, retry_after=None), 2)

    asyncio.run(scenario())

    assert time.slept == [5.0, 10.0]


def test_tokens_are_estimated_from_the_prompt() -> None:
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="a" * 400)])]

    assert estimate_tokens(messages) == 100 + 900


def refusing_model(failures: int, status: int = 429) -> tuple[FunctionModel, list[int]]:
    calls: list[int] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(len(calls) + 1)
        if len(calls) <= failures:
            raise refusal(status=status, retry_after="9")
        return ModelResponse(parts=[TextPart("ok")])

    return FunctionModel(respond), calls


def test_a_full_provider_is_waited_out_rather_than_abandoned() -> None:
    pacer, time = a_pacer(Limits(30, 8_000, 1_000, 200_000))
    model, calls = refusing_model(failures=2)

    result = asyncio.run(Agent(PacedModel(model, pacer)).run("write a question"))

    assert result.output == "ok"
    # The same provider answered on the third try, so no other provider was spent.
    assert len(calls) == 3
    assert time.slept == [11.0, 11.0]


def test_a_provider_that_keeps_refusing_lets_the_error_through() -> None:
    pacer, time = a_pacer(Limits(30, 8_000, 1_000, 200_000))
    model, calls = refusing_model(failures=5)

    with pytest.raises(ModelHTTPError):
        asyncio.run(Agent(PacedModel(model, pacer, attempts=3)).run("write a question"))

    assert len(calls) == 3
    assert time.slept == [11.0, 11.0]


def test_a_refusal_that_will_not_pass_is_raised_at_once() -> None:
    pacer, time = a_pacer(Limits(30, 8_000, 1_000, 200_000))
    model, calls = refusing_model(failures=5, status=400)

    with pytest.raises(ModelHTTPError):
        asyncio.run(Agent(PacedModel(model, pacer)).run("write a question"))

    assert (len(calls), time.slept) == (1, [])

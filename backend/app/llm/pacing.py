"""Holding each provider to its free tier, and riding out its refusals.

A 429 is not a reason to take the work elsewhere. Groq's token bucket refills at about 133
tokens a second, so waiting out the delay it asks for costs seconds, while falling straight
through to the next provider spends a second daily allowance the day may still need. Every
model is therefore wrapped in a pacer that holds a request until the provider has room for
it, waits out a refusal with a margin on top of the delay the provider named, and only lets
the error through -- so that a fallback chain moves on -- once the provider is out of
attempts or out of budget for the day.

The trial measured what this is built on: Groq answers 30 requests and 8,000 tokens a minute
on the free tier, its 429 names the tokens it wanted (the prompt plus 400 to 900, with
max_tokens not reserved), and twice the retry-after it sent was too short to succeed on.
Some models are also held to a ceiling on the tokens they write a minute: Groq caps Qwen 3.8
at 1,000, and names that limit only in its 429s.
"""

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

log = logging.getLogger(__name__)

WINDOW = 60.0
# What a provider charges for the answer on top of the prompt, read off the trial's 429 bodies
RESPONSE_ALLOWANCE = 900
# Added to whatever delay the provider asks for: twice in the trial its own figure was short
RETRY_MARGIN = 2.0
# Used when a refusal names no delay, doubling per attempt
BACKOFF = 5.0
RETRYABLE = frozenset({429, 500, 502, 503, 504})

Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class Limits:
    requests_per_minute: int
    tokens_per_minute: int
    requests_per_day: int
    tokens_per_day: int
    # Only some models have one; None means the provider does not count output separately.
    output_tokens_per_minute: int | None = None


class QuotaExhausted(ModelAPIError):
    """The provider has nothing left for today. Raised as a model error so that a fallback
    chain treats it like any other reason to try the next provider."""


def estimate_tokens(messages: list[ModelMessage]) -> int:
    """Roughly what a request will cost, to keep inside a tokens-a-minute ceiling.

    Only the order of magnitude matters, so the prompt is counted at four characters to the
    token and the answer is allowed for at the top of the range the trial saw.
    """
    characters = sum(
        len(str(getattr(part, "content", "") or ""))
        for message in messages
        for part in message.parts
    )
    return characters // 4 + RESPONSE_ALLOWANCE


def asked_delay(exc: ModelHTTPError) -> float | None:
    """How long the provider asked us to wait, from its header or from its message."""
    header = (exc.headers or {}).get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    found = re.search(r"try again in ([\d.]+)\s*s", str(exc.body or ""))
    return float(found.group(1)) if found else None


class Pacer:
    """One provider's budget: a minute window for its rates and a running total for its day.

    The day starts from what was already spent in it: a pacer is built for every batch, and
    without that a second batch in a day would think it had the whole day to itself.
    """

    def __init__(
        self,
        name: str,
        limits: Limits,
        *,
        spent: tuple[int, int] = (0, 0),
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
        margin: float = RETRY_MARGIN,
    ) -> None:
        self.name = name
        self.limits = limits
        self._clock = clock
        self._sleep = sleep
        self._margin = margin
        # [when it started, tokens it cost, tokens it wrote] per request in the last minute
        self._window: list[list[float]] = []
        self._requests_today, self._tokens_today = spent
        self._blocked_until = 0.0

    @property
    def spent(self) -> tuple[int, int]:
        """Requests and tokens spent today."""
        return self._requests_today, self._tokens_today

    async def acquire(self, estimate: int) -> list[float]:
        """Wait until the provider has room, then book the request in. Returns its entry."""
        while True:
            now = self._clock()
            self._window = [entry for entry in self._window if entry[0] > now - WINDOW]
            if self._requests_today >= self.limits.requests_per_day:
                raise QuotaExhausted(self.name, f"{self.name} is out of requests for today")
            if self._tokens_today + estimate > self.limits.tokens_per_day:
                raise QuotaExhausted(self.name, f"{self.name} is out of tokens for today")

            waits = [self._blocked_until - now]
            requests = len(self._window)
            tokens = sum(entry[1] for entry in self._window)
            written = sum(entry[2] for entry in self._window)
            output_limit = self.limits.output_tokens_per_minute
            if (
                requests + 1 > self.limits.requests_per_minute
                or tokens + estimate > self.limits.tokens_per_minute
                or (output_limit is not None and written + RESPONSE_ALLOWANCE > output_limit)
            ):
                waits.append(self._window[0][0] + WINDOW - now if self._window else 0.0)
            if (wait := max(waits)) > 0:
                log.info("%s: waiting %.1f s for room", self.name, wait)
                await self._sleep(wait)
                continue

            # The answer is booked at the allowance until its real length is known.
            entry = [now, float(estimate), float(RESPONSE_ALLOWANCE)]
            self._window.append(entry)
            self._requests_today += 1
            self._tokens_today += estimate
            return entry

    def record(self, entry: list[float], tokens: int, output: int = 0) -> None:
        """Correct the estimates booked by `acquire` once the real cost is known."""
        self._tokens_today += tokens - int(entry[1])
        entry[1] = float(tokens)
        entry[2] = float(output)

    async def wait_out(self, exc: ModelHTTPError, attempt: int) -> float:
        """Sleep off a refusal, and hold anything else back for as long."""
        named = asked_delay(exc)
        delay = (named + self._margin) if named is not None else BACKOFF * 2 ** (attempt - 1)
        self._blocked_until = self._clock() + delay
        log.info(
            "%s: HTTP %s on attempt %d, waiting %.1f s%s",
            self.name,
            exc.status_code,
            attempt,
            delay,
            f" ({named} s asked for)" if named is not None else "",
        )
        await self._sleep(delay)
        return delay


class PacedModel(WrapperModel):
    """A model that keeps its provider's pace and rides out its retryable refusals."""

    def __init__(self, wrapped: Model, pacer: Pacer, *, attempts: int = 3) -> None:
        super().__init__(wrapped)
        self.pacer = pacer
        self.attempts = attempts

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        for attempt in range(1, self.attempts + 1):
            entry = await self.pacer.acquire(estimate_tokens(messages))
            try:
                response = await super().request(messages, model_settings, model_request_parameters)
            except ModelHTTPError as exc:
                self.pacer.record(entry, 0)
                if attempt == self.attempts or exc.status_code not in RETRYABLE:
                    raise
                await self.pacer.wait_out(exc, attempt)
                continue
            usage = response.usage
            self.pacer.record(entry, usage.input_tokens + usage.output_tokens, usage.output_tokens)
            return response
        raise AssertionError("unreachable")

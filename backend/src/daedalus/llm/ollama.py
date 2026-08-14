"""
The real language model: Ollama over HTTP.

Ollama exposes an OpenAI-shaped chat endpoint on localhost, so this is a
plain HTTP client — no vendor SDK, and no new dependency, since httpx is
already used elsewhere.

Streaming is switched off. The API returns one JSON object per response
that way, and nothing downstream consumes tokens incrementally yet; the
endpoint that serves this waits for the whole answer regardless.
"""

from __future__ import annotations

import logging
import re

import httpx

from daedalus.config import settings
from daedalus.core.exceptions import LLMError
from daedalus.interfaces.llm import LLM

__all__ = ["OllamaLLM"]


logger = logging.getLogger(__name__)


# Reasoning models such as qwen3 wrap their deliberation in <think> tags
# and return it inside the same content field as the answer. It is not part
# of the answer, it frequently contradicts the answer, and it would be
# scored as if it were the answer by the evaluation harness.
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove reasoning blocks a model emits alongside its answer."""

    return _THINK.sub("", text).strip()


class OllamaLLM(LLM):
    """Generates with a model served by a local Ollama instance."""

    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> None:
        self._model = model or settings.llm_model
        self._base_url = (base_url or settings.ollama_url).rstrip("/")
        self._temperature = settings.llm_temperature if temperature is None else temperature
        self._timeout = settings.llm_timeout_seconds if timeout is None else timeout

    @property
    def model(self) -> str:
        return self._model

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        messages = []

        if system is not None:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self._temperature},
        }

        try:
            response = httpx.post(f"{self._base_url}/api/chat", json=payload, timeout=self._timeout)
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            # A 404 here almost always means the model was never pulled,
            # which is a different fix from the server being down.
            raise LLMError(
                f"Ollama rejected the request for model {self._model!r} "
                f"({error.response.status_code}). Try: ollama pull {self._model}"
            ) from error
        except httpx.HTTPError as error:
            raise LLMError(
                f"could not reach Ollama at {self._base_url}. Is `ollama serve` running?"
            ) from error

        content = response.json().get("message", {}).get("content", "")
        answer = strip_reasoning(content)

        # A model that returns only a reasoning block leaves nothing behind
        # once it is stripped. Silently returning "" would look to the
        # caller like a confident empty answer.
        if not answer:
            raise LLMError(f"{self._model} returned no answer text")

        return answer

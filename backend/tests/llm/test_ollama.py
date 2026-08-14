"""
The Ollama adapter, tested against a stubbed transport.

No server is contacted: httpx.post is replaced, so the failure paths — the
ones that matter and are hardest to reproduce on demand — can be exercised
deterministically.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from daedalus.core.exceptions import LLMError
from daedalus.llm.ollama import OllamaLLM, strip_reasoning


class _Response:
    """Just enough of httpx.Response for the adapter."""

    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)  # type: ignore[arg-type]

    def json(self) -> dict[str, Any]:
        return self._payload


def _reply(text: str) -> _Response:
    return _Response({"message": {"content": text}})


# Reasoning Removal


def test_a_reasoning_block_is_stripped() -> None:
    assert strip_reasoning("<think>hmm, maybe</think>The answer is 4.") == "The answer is 4."


def test_a_multiline_reasoning_block_is_stripped() -> None:
    text = "<think>\nstep one\nstep two\n</think>\nFinal answer."

    assert strip_reasoning(text) == "Final answer."


def test_text_without_reasoning_is_untouched() -> None:
    assert strip_reasoning("A plain answer.") == "A plain answer."


def test_reasoning_is_stripped_from_a_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    """qwen3 returns deliberation in the same field as the answer."""

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _reply("<think>x</think>Scaled."))

    assert OllamaLLM().complete("why?") == "Scaled."


# Requests


def test_the_system_prompt_is_sent_separately(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _Response:
        captured.update(kwargs["json"])
        captured["url"] = url
        return _reply("ok")

    monkeypatch.setattr(httpx, "post", fake_post)

    OllamaLLM(model="test-model").complete("the question", system="the rules")

    assert captured["url"].endswith("/api/chat")
    assert captured["model"] == "test-model"
    assert captured["stream"] is False
    assert captured["messages"] == [
        {"role": "system", "content": "the rules"},
        {"role": "user", "content": "the question"},
    ]


def test_no_system_message_is_sent_when_none_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        httpx, "post", lambda url, **kw: (captured.update(kw["json"]), _reply("ok"))[1]
    )

    OllamaLLM().complete("just the question")

    assert [message["role"] for message in captured["messages"]] == ["user"]


def test_a_trailing_slash_on_the_url_does_not_double_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _Response:
        captured["url"] = url
        return _reply("ok")

    monkeypatch.setattr(httpx, "post", fake_post)

    OllamaLLM(base_url="http://localhost:11434/").complete("q")

    assert captured["url"] == "http://localhost:11434/api/chat"


# Failures


def test_an_unreachable_server_is_reported_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> _Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", refuse)

    with pytest.raises(LLMError, match="ollama serve"):
        OllamaLLM().complete("q")


def test_a_missing_model_suggests_pulling_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Response({}, status_code=404))

    with pytest.raises(LLMError, match="ollama pull"):
        OllamaLLM(model="never-pulled").complete("q")


def test_an_empty_answer_is_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _reply(""))

    with pytest.raises(LLMError, match="no answer"):
        OllamaLLM().complete("q")


def test_an_answer_that_is_only_reasoning_is_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stripping must not turn a non-answer into a silent empty string."""

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _reply("<think>still thinking</think>"))

    with pytest.raises(LLMError, match="no answer"):
        OllamaLLM().complete("q")

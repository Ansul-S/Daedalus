"""Tests for the Ollama embedding client.

No test contacts a live model: the HTTP layer is replaced with a stub.
"""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Any

import pytest

from daedalus.embedding import (
    DEFAULT_OLLAMA_URL,
    OLLAMA_URL_ENV,
    EmbeddingError,
    embed_texts,
    ollama_url,
)


class FakeResponse(io.BytesIO):
    """Minimal stand-in for the object urlopen returns."""

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def stub_urlopen(
    monkeypatch: pytest.MonkeyPatch, body: object, captured: dict[str, Any]
) -> None:
    def fake(request: Any, timeout: float = 0) -> FakeResponse:
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse(json.dumps(body).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake)


def test_ollama_url_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(OLLAMA_URL_ENV, raising=False)
    assert ollama_url() == DEFAULT_OLLAMA_URL


def test_ollama_url_honours_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OLLAMA_URL_ENV, "http://elsewhere:9999")
    assert ollama_url() == "http://elsewhere:9999"


def test_empty_input_makes_no_request(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("no request should be made")

    monkeypatch.setattr("urllib.request.urlopen", explode)
    assert embed_texts([]) == []


def test_returns_one_vector_per_input(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    stub_urlopen(monkeypatch, {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}, captured)

    vectors = embed_texts(["a", "b"], model="bge-m3")

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert captured["payload"] == {"model": "bge-m3", "input": ["a", "b"]}
    assert captured["url"].endswith("/api/embed")


def test_uses_the_configured_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    stub_urlopen(monkeypatch, {"embeddings": [[1.0]]}, captured)

    embed_texts(["a"], url="http://somewhere:1234")

    assert captured["url"] == "http://somewhere:1234/api/embed"


def test_unreachable_server_raises_embedding_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", refuse)

    with pytest.raises(EmbeddingError, match="could not reach Ollama"):
        embed_texts(["a"])


def test_wrong_vector_count_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    stub_urlopen(monkeypatch, {"embeddings": [[1.0]]}, captured)

    with pytest.raises(EmbeddingError, match="expected 2 vectors"):
        embed_texts(["a", "b"])


def test_missing_embeddings_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    stub_urlopen(monkeypatch, {"error": "model not found"}, captured)

    with pytest.raises(EmbeddingError):
        embed_texts(["a"])

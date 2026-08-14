"""The /answer endpoint, over material ingested through the real pipeline."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from daedalus.api.dependencies import get_llm
from daedalus.api.main import app
from daedalus.core.exceptions import LLMError
from daedalus.generation.prompts import REFUSAL
from daedalus.interfaces.llm import LLM
from daedalus.llm import FakeLLM

Upload = Callable[..., httpx.Response]


@pytest.fixture
def ingested(upload: Upload, document: Path, api: TestClient) -> TestClient:
    upload(document)

    return api


def test_a_question_is_answered(ingested: TestClient) -> None:
    response = ingested.post("/answer", json={"question": "what is layer normalization?"})

    assert response.status_code == 200
    assert response.json()["answer"]


def test_the_answer_carries_its_citations(ingested: TestClient) -> None:
    body = ingested.post("/answer", json={"question": "softmax scaling"}).json()

    assert body["citations"]
    assert body["citations"][0]["doc_id"] == "attention"
    assert body["citations"][0]["source_end"] > body["citations"][0]["source_start"]


def test_sources_are_reported_alongside_citations(ingested: TestClient) -> None:
    """The gap between the two is what separates a retrieval failure from a grounding one."""

    body = ingested.post("/answer", json={"question": "softmax scaling"}).json()

    assert len(body["sources"]) >= len(body["citations"])


def test_the_model_is_reported(ingested: TestClient) -> None:
    body = ingested.post("/answer", json={"question": "softmax"}).json()

    assert body["model"] == "fake"


def test_an_unanswerable_question_refuses_without_citations(api: TestClient) -> None:
    """Refusing correctly is a 200 — it is the right answer, not an error."""

    body = api.post("/answer", json={"question": "how does LoRA save memory?"}).json()

    assert body["answer"] == REFUSAL
    assert body["citations"] == []


def test_an_empty_question_is_rejected(ingested: TestClient) -> None:
    assert ingested.post("/answer", json={"question": ""}).status_code == 422


def test_an_out_of_range_top_k_is_rejected(ingested: TestClient) -> None:
    assert ingested.post("/answer", json={"question": "x", "top_k": 0}).status_code == 422
    assert ingested.post("/answer", json={"question": "x", "top_k": 99}).status_code == 422


def test_an_unavailable_model_is_reported_as_unavailable(ingested: TestClient) -> None:
    """A backend that is down is transient, and not the client's fault."""

    class DownLLM(FakeLLM):
        def complete(self, prompt: str, *, system: str | None = None) -> str:
            raise LLMError("could not reach Ollama")

    def down() -> LLM:
        return DownLLM()

    app.dependency_overrides[get_llm] = down

    try:
        response = ingested.post("/answer", json={"question": "softmax"})
    finally:
        del app.dependency_overrides[get_llm]

    assert response.status_code == 503

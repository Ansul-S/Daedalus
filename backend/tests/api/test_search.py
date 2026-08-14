"""The search endpoint, over material ingested through the real pipeline."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

Upload = Callable[..., httpx.Response]


@pytest.fixture
def ingested(upload: Upload, document: Path, api: TestClient) -> TestClient:
    """A client with one document indexed."""

    upload(document)

    return api


def test_a_query_returns_results(ingested: TestClient) -> None:
    response = ingested.post("/search", json={"query": "layer normalization"})

    assert response.status_code == 200
    assert response.json()["results"]


def test_results_carry_the_text_and_its_source_offsets(ingested: TestClient) -> None:
    """Offsets are what a citation resolves to."""

    result = ingested.post("/search", json={"query": "softmax"}).json()["results"][0]

    assert result["text"]
    assert result["doc_id"] == "attention"
    assert result["source_end"] > result["source_start"]
    assert result["extraction"] in {"text", "ocr", "vision", "notebook"}


def test_results_are_ordered_best_first(ingested: TestClient) -> None:
    results = ingested.post("/search", json={"query": "normalization", "top_k": 3}).json()[
        "results"
    ]

    scores = [result["score"] for result in results]

    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_the_results(ingested: TestClient) -> None:
    results = ingested.post("/search", json={"query": "the", "top_k": 1}).json()["results"]

    assert len(results) == 1


def test_the_query_is_echoed_back(ingested: TestClient) -> None:
    body = ingested.post("/search", json={"query": "softmax"}).json()

    assert body["query"] == "softmax"


def test_searching_an_empty_index_is_not_an_error(api: TestClient) -> None:
    response = api.post("/search", json={"query": "anything at all"})

    assert response.status_code == 200
    assert response.json()["results"] == []


def test_a_hostile_query_does_not_reach_fts5(ingested: TestClient) -> None:
    """Raw, this is a syntax error inside MATCH."""

    response = ingested.post("/search", json={"query": 'C++ "quoted" NEAR(a b)'})

    assert response.status_code == 200


def test_an_empty_query_is_rejected(ingested: TestClient) -> None:
    assert ingested.post("/search", json={"query": ""}).status_code == 422


def test_an_out_of_range_top_k_is_rejected(ingested: TestClient) -> None:
    assert ingested.post("/search", json={"query": "softmax", "top_k": 0}).status_code == 422
    assert ingested.post("/search", json={"query": "softmax", "top_k": 999}).status_code == 422

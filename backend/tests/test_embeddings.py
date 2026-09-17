import asyncio
import json

import httpx
import pytest

from app.core.config import Settings
from app.llm.embeddings import BATCH_SIZE, Embedder, EmbeddingError, query_input


def embedder_for(handler) -> Embedder:
    settings = Settings(_env_file=None, embedding_model="embed-test", embedding_num_ctx=1024)
    http = httpx.AsyncClient(base_url="http://ollama", transport=httpx.MockTransport(handler))
    return Embedder(settings, http=http)


def vectors_for(request: httpx.Request, dimensions: int = 1024) -> httpx.Response:
    inputs = json.loads(request.content)["input"]
    return httpx.Response(200, json={"embeddings": [[0.5] * dimensions for _ in inputs]})


async def embed_documents(embedder: Embedder, texts: list[str]) -> list[list[float]]:
    async with embedder:
        return await embedder.embed_documents(texts)


def test_documents_are_sent_in_batches_with_a_fixed_context() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return vectors_for(request)

    texts = [f"text {n}" for n in range(BATCH_SIZE * 2 + 1)]
    vectors = asyncio.run(embed_documents(embedder_for(handler), texts))

    assert len(vectors) == len(texts)
    assert [len(body["input"]) for body in requests] == [BATCH_SIZE, BATCH_SIZE, 1]
    for body in requests:
        assert body["model"] == "embed-test"
        assert body["truncate"] is False
        assert body["options"] == {"num_ctx": 1024}


def test_only_queries_get_the_instruction() -> None:
    inputs: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        inputs.extend(json.loads(request.content)["input"])
        return vectors_for(request)

    async def run() -> None:
        async with embedder_for(handler) as embedder:
            await embedder.embed_query("why scale by sqrt(d_k)?")
            await embedder.embed_documents(["a passage"])

    asyncio.run(run())
    assert inputs == [query_input("why scale by sqrt(d_k)?"), "a passage"]
    assert inputs[0].startswith("Instruct: ") and inputs[0].endswith(
        "\nQuery:why scale by sqrt(d_k)?"
    )


@pytest.mark.parametrize(
    "handler",
    [
        lambda request: httpx.Response(400, json={"error": "input length exceeds context"}),
        lambda request: vectors_for(request, dimensions=768),
        lambda request: httpx.Response(200, json={"embeddings": []}),
    ],
    ids=["http-error", "wrong-dimensions", "missing-vectors"],
)
def test_bad_responses_raise_a_clear_error(handler) -> None:
    with pytest.raises(EmbeddingError):
        asyncio.run(embed_documents(embedder_for(handler), ["text"]))


def test_an_unreachable_server_is_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(EmbeddingError, match="make ollama"):
        asyncio.run(embed_documents(embedder_for(handler), ["text"]))

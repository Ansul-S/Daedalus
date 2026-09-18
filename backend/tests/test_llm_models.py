import asyncio
import json

# The OpenAI SDK that talks to Ollama is built on httpx2, so its transports are mocked here.
import httpx2
import pytest
from pydantic import SecretStr
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel

from app.core.config import Settings
from app.llm.models import generation_model, grading_model, helper_model, helper_settings

KEY = SecretStr("test-key")
COMPLETION = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 0,
    "model": "qwen3.5:4b",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def make_settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_without_cloud_keys_everything_runs_locally() -> None:
    settings = make_settings(groq_api_key=None, gemini_api_key=None)
    assert generation_model(settings).system == "ollama"
    assert grading_model(settings).system == "ollama"


def test_generation_prefers_cloud_and_grading_prefers_local() -> None:
    settings = make_settings(groq_api_key=KEY, gemini_api_key=KEY)

    generation = generation_model(settings)
    grading = grading_model(settings)

    assert isinstance(generation, FallbackModel)
    assert isinstance(grading, FallbackModel)
    assert [model.system for model in generation.models] == ["groq", "google", "ollama"]
    assert [model.system for model in grading.models] == ["ollama", "groq", "google"]


def test_production_never_uses_ollama() -> None:
    settings = make_settings(environment="production", groq_api_key=KEY, gemini_api_key=None)
    assert grading_model(settings).system == "groq"
    assert generation_model(settings).system == "groq"


def test_production_without_keys_fails_clearly() -> None:
    settings = make_settings(environment="production", groq_api_key=None, gemini_api_key=None)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        grading_model(settings)


def test_the_local_helper_asks_ollama_not_to_think() -> None:
    """Ollama switches thinking on by itself for a model that can think, which turns a nine
    second call into a four minute one. The request has to say otherwise, and a library that
    stops passing the setting on must fail here rather than in a batch run."""
    bodies = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json=COMPLETION)

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    model = helper_model(make_settings(), client)

    result = asyncio.run(Agent(model).run("tag this passage", model_settings=helper_settings()))

    assert result.output == "ok"
    [body] = bodies
    assert body["reasoning_effort"] == "none"
    # Ollama's OpenAI-compatible endpoint samples at 1.0 for anything it is not sent.
    assert (body["temperature"], body["top_p"], body["seed"]) == (0.0, 1.0, 7)
    assert body["model"] == "qwen3.5:4b"


def test_the_helper_is_not_available_in_production() -> None:
    with pytest.raises(RuntimeError, match="Ollama"):
        helper_model(make_settings(environment="production"))

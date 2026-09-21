import asyncio
import json

# The OpenAI SDK that talks to Ollama is built on httpx2, so its transports are mocked here;
# the Groq SDK is still on httpx.
import httpx
import httpx2
import pytest
from pydantic import BaseModel, SecretStr
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models.fallback import FallbackModel

from app.core.config import Settings
from app.llm.models import (
    generation_model,
    grading_model,
    groq,
    helper_model,
    helper_settings,
    paced_generation_model,
)

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


def test_generation_and_grading_both_prefer_groq() -> None:
    settings = make_settings(groq_api_key=KEY, gemini_api_key=KEY)

    generation = generation_model(settings)
    grading = grading_model(settings)

    assert isinstance(generation, FallbackModel)
    assert isinstance(grading, FallbackModel)
    assert [model.system for model in generation.models] == ["groq", "google", "ollama"]
    assert [model.system for model in grading.models] == ["groq", "ollama", "google"]


def test_the_generator_never_grades() -> None:
    """A model family does not grade answers to its own questions, fallback included."""
    settings = make_settings(groq_api_key=KEY, gemini_api_key=KEY)

    names = [model.model_name for model in grading_model(settings).models]

    assert settings.groq_model not in names
    assert names[0] == "qwen/qwen3.8-27b"


def test_grading_starts_each_provider_from_its_spend_today() -> None:
    settings = make_settings(groq_api_key=KEY, gemini_api_key=KEY)

    chain = grading_model(settings, {settings.groq_grading_model: (3, 5_000)})

    qwen, _, gemini = chain.models
    assert (qwen.pacer.spent, gemini.pacer.spent) == ((3, 5_000), (0, 0))
    assert qwen.pacer.limits.output_tokens_per_minute == 1_000


def test_a_paced_batch_starts_each_provider_from_its_spend_today() -> None:
    settings = make_settings(groq_api_key=KEY, gemini_api_key=KEY)

    chain = paced_generation_model(settings, {settings.groq_model: (12, 45_000)})

    groq, gemini, _ = chain.models
    assert (groq.pacer.spent, gemini.pacer.spent) == ((12, 45_000), (0, 0))


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


class Verdict(BaseModel):
    status: str


def test_the_grader_on_groq_gets_a_strict_schema_and_does_not_think() -> None:
    """The profile Pydantic AI picks for Qwen on Groq refuses structured output, although
    Groq serves it; a library change that undoes the fix must fail here."""
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        return httpx.Response(
            200,
            json=COMPLETION
            | {
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"status": "covered"}'},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    settings = make_settings(groq_api_key=KEY)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    model = groq(settings, settings.groq_grading_model, client)
    agent = Agent(model, output_type=NativeOutput(Verdict, strict=True))

    result = asyncio.run(agent.run("grade this", model_settings={"thinking": False}))

    assert result.output == Verdict(status="covered")
    [body] = bodies
    assert body["model"] == "qwen/qwen3.8-27b"
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["reasoning_effort"] == "none"


def test_the_helper_is_not_available_in_production() -> None:
    with pytest.raises(RuntimeError, match="Ollama"):
        helper_model(make_settings(environment="production"))

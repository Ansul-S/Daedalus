"""Which model handles which task, and what each task falls back to.

Question generation is bulk work, so it prefers the free cloud tiers and keeps the local
model as a last resort. Grading is interactive: it goes to Qwen on Groq first, which grades
in about two seconds where the local model on a 16 GB machine takes a minute or two, then to
the local model, then to Gemini. The generator is gpt-oss, so grading never falls back to
it -- a model family does not grade its own questions. Tagging chunks and checking that a
question is answerable are light local work with no fallback. In production there is no
Ollama, so only cloud models are used.
"""

from collections.abc import Mapping
from typing import Any

from groq import AsyncGroq
from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.profiles import ModelProfile, merge_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.settings import ModelSettings

from app.core.config import Settings
from app.llm.pacing import Limits, PacedModel, Pacer

# Ollama serves the same output for the same prompt at this seed.
HELPER_SEED = 7

# Measured on the free tier in the trial
GROQ_FREE = Limits(
    requests_per_minute=30,
    tokens_per_minute=8_000,
    requests_per_day=1_000,
    tokens_per_day=200_000,
)
# Qwen on Groq has the same limits, plus one on what it writes that Groq names only in its 429s
GROQ_QWEN_FREE = Limits(
    requests_per_minute=30,
    tokens_per_minute=8_000,
    requests_per_day=1_000,
    tokens_per_day=200_000,
    output_tokens_per_minute=1_000,
)
# Requests a day as Gemini's 429 states it for the free tier; the minute rates are not
# measured and are held low so the pacer errs towards waiting rather than being refused.
GEMINI_FREE = Limits(
    requests_per_minute=10,
    tokens_per_minute=100_000,
    requests_per_day=20,
    tokens_per_day=1_000_000,
)


def ollama(settings: Settings, name: str, http_client: Any = None) -> OllamaModel | None:
    """A model served by the local Ollama.

    Ollama turns thinking on by itself for any model that can think, and qwen3.5 can. The
    profile Pydantic AI picks for it does not declare thinking support, so a unified
    `thinking` setting is dropped before the request is built and the model thinks anyway --
    minutes per call instead of seconds. Declaring the support here lets `thinking=False`
    through as `reasoning_effort: "none"`, which Ollama reads as think=false.
    """
    if settings.environment != "local":
        return None
    provider = OllamaProvider(base_url=f"{settings.ollama_base_url}/v1", http_client=http_client)
    profile = merge_profile(provider.model_profile(name), ModelProfile(supports_thinking=True))
    return OllamaModel(name, provider=provider, profile=profile)


def helper_settings() -> ModelSettings:
    """Settings for every call to the small local model: no thinking, and the same answer
    every time. Ollama's OpenAI-compatible endpoint defaults temperature and top_p to 1.0
    when they are not sent, so both are sent."""
    return ModelSettings(thinking=False, temperature=0.0, top_p=1.0, seed=HELPER_SEED)


def groq(settings: Settings, name: str | None = None, http_client: Any = None) -> GroqModel | None:
    """Groq, the generation model unless another is named, with the client's own retrying
    turned off.

    Left on, it sleeps through a 429 twice before anything else sees the error, so our pacer
    never learns the provider is full and never gets to apply its own margin.

    Groq constrains every model used here to a strict JSON schema, but the profile Pydantic
    AI picks for Qwen does not say so and refuses structured output outright. Declaring it
    changes nothing for gpt-oss, whose profile already does.
    """
    if settings.groq_api_key is None:
        return None
    client = AsyncGroq(
        api_key=settings.groq_api_key.get_secret_value(), max_retries=0, http_client=http_client
    )
    provider = GroqProvider(groq_client=client)
    name = name or settings.groq_model
    profile = merge_profile(
        provider.model_profile(name),
        ModelProfile(
            supports_json_schema_output=True, json_schema_transformer=OpenAIJsonSchemaTransformer
        ),
    )
    return GroqModel(name, provider=provider, profile=profile)


def gemini(settings: Settings) -> GoogleModel | None:
    if settings.gemini_api_key is None:
        return None
    provider = GoogleProvider(api_key=settings.gemini_api_key.get_secret_value())
    return GoogleModel(settings.gemini_model, provider=provider)


def _chain(*candidates: Model | None) -> Model:
    models = [model for model in candidates if model is not None]
    if not models:
        raise RuntimeError("No model available: set GROQ_API_KEY or GEMINI_API_KEY")
    return models[0] if len(models) == 1 else FallbackModel(*models)


def generation_model(settings: Settings) -> Model:
    """Question generation: Groq, then Gemini, then the local grader model."""
    return _chain(groq(settings), gemini(settings), ollama(settings, settings.grader_model))


def grading_model(settings: Settings, spent: Mapping[str, tuple[int, int]] | None = None) -> Model:
    """Answer grading: Qwen on Groq, then the local grader model, then Gemini.

    Both cloud models are paced; `spent` is what each has used today, by model name, as for
    generation.
    """
    spent = spent or {}
    name = settings.groq_grading_model
    return _chain(
        _paced(groq(settings, name), "groq-grading", GROQ_QWEN_FREE, spent.get(name)),
        ollama(settings, settings.grader_model),
        _paced(gemini(settings), "gemini", GEMINI_FREE, spent.get(settings.gemini_model)),
    )


def paced_generation_model(
    settings: Settings, spent: Mapping[str, tuple[int, int]] | None = None
) -> Model:
    """Question generation in bulk: Groq, then Gemini, then the local grader.

    Each cloud model keeps its own pace, so a provider that is full for the moment is waited
    out rather than abandoned; the chain moves on only once a provider has run out of
    attempts or out of the day's budget. `spent` is the requests and tokens each model has
    already used today, by model name, so that the day's budget is the day's and not the
    batch's.
    """
    spent = spent or {}
    return _chain(
        _paced(groq(settings), "groq", GROQ_FREE, spent.get(settings.groq_model)),
        _paced(gemini(settings), "gemini", GEMINI_FREE, spent.get(settings.gemini_model)),
        ollama(settings, settings.grader_model),
    )


def _paced(
    model: Model | None, name: str, limits: Limits, spent: tuple[int, int] | None
) -> Model | None:
    if model is None:
        return None
    return PacedModel(model, Pacer(name, limits, spent=spent or (0, 0)))


def helper_model(settings: Settings, http_client: Any = None) -> OllamaModel:
    """Tagging chunks and checking answerability: the small local model only. Both jobs run
    over the whole library, so they stay off the cloud quotas."""
    model = ollama(settings, settings.helper_model, http_client)
    if model is None:
        raise RuntimeError(f"{settings.helper_model} needs Ollama, which only runs locally")
    return model

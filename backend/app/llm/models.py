"""Which model handles which task, and what each task falls back to.

Question generation is bulk work, so it prefers the free cloud tiers and keeps the local
model as a last resort. Grading is interactive and handles the user's own answers, so it
prefers the local model and falls back to Groq, then Gemini. Tagging chunks and checking
that a question is answerable are light local work with no fallback. In production there is
no Ollama, so only cloud models are used.
"""

from typing import Any

from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.profiles import ModelProfile, merge_profile
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.settings import ModelSettings

from app.core.config import Settings

# Ollama serves the same output for the same prompt at this seed.
HELPER_SEED = 7


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


def groq(settings: Settings) -> GroqModel | None:
    if settings.groq_api_key is None:
        return None
    provider = GroqProvider(api_key=settings.groq_api_key.get_secret_value())
    return GroqModel(settings.groq_model, provider=provider)


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


def grading_model(settings: Settings) -> Model:
    """Answer grading: the local grader model, then Groq, then Gemini."""
    return _chain(ollama(settings, settings.grader_model), groq(settings), gemini(settings))


def helper_model(settings: Settings, http_client: Any = None) -> OllamaModel:
    """Tagging chunks and checking answerability: the small local model only. Both jobs run
    over the whole library, so they stay off the cloud quotas."""
    model = ollama(settings, settings.helper_model, http_client)
    if model is None:
        raise RuntimeError(f"{settings.helper_model} needs Ollama, which only runs locally")
    return model

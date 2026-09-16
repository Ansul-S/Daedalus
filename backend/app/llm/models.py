"""Which model handles which task, and what each task falls back to.

Question generation is bulk work, so it prefers the free cloud tiers and keeps the local
model as a last resort. Grading is interactive and handles the user's own answers, so it
prefers the local model and falls back to Groq, then Gemini. In production there is no
Ollama, so only cloud models are used.
"""

from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.providers.ollama import OllamaProvider

from app.core.config import Settings


def ollama(settings: Settings, name: str) -> OllamaModel | None:
    if settings.environment != "local":
        return None
    return OllamaModel(name, provider=OllamaProvider(base_url=f"{settings.ollama_base_url}/v1"))


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

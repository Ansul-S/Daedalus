import pytest
from pydantic import SecretStr
from pydantic_ai.models.fallback import FallbackModel

from app.core.config import Settings
from app.llm.models import generation_model, grading_model

KEY = SecretStr("test-key")


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

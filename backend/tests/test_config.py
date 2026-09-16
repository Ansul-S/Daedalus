import pytest

from app.core.checks import check_cloud_keys, is_installed
from app.core.config import Settings


def test_blank_api_keys_count_as_unset(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    env_file = tmp_path / "test.env"
    env_file.write_text("GROQ_API_KEY=\nGEMINI_API_KEY=\n")

    settings = Settings(_env_file=env_file)

    assert settings.groq_api_key is None
    assert settings.gemini_api_key is None


def test_missing_keys_are_warnings_locally() -> None:
    settings = Settings(_env_file=None, groq_api_key=None, gemini_api_key=None)
    assert {check.status for check in check_cloud_keys(settings)} == {"warn"}


def test_production_needs_at_least_one_key() -> None:
    settings = Settings(
        _env_file=None, environment="production", groq_api_key=None, gemini_api_key=None
    )
    assert {check.status for check in check_cloud_keys(settings)} == {"fail"}


def test_untagged_model_names_match_latest() -> None:
    assert is_installed("qwen3.5", {"qwen3.5:latest"})
    assert is_installed("qwen3.5:9b", {"qwen3.5:9b"})
    assert not is_installed("qwen3.5:9b", {"qwen3.5:4b"})

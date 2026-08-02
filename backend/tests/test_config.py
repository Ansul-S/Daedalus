"""Tests for configuration loading and the constants/settings split."""

from __future__ import annotations

import pytest

from daedalus.config import constants
from daedalus.config.settings import BACKEND_ROOT, Settings


def test_defaults_are_applied() -> None:
    config = Settings()

    assert config.app_name == "Daedalus"
    assert config.debug is False
    assert config.log_level == "INFO"


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAEDALUS_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DAEDALUS_PORT", "9000")

    config = Settings()

    assert config.log_level == "DEBUG"
    assert config.port == 9000


def test_env_prefix_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unprefixed variable must not leak into settings."""

    monkeypatch.setenv("LOG_LEVEL", "CRITICAL")

    assert Settings().log_level == "INFO"


def test_backend_root_resolves_to_the_backend_directory() -> None:
    """
    Paths are anchored to the source tree, not the working directory.

    This is what lets the app be launched from anywhere without silently
    losing its .env file or writing data to the wrong place.
    """

    assert BACKEND_ROOT.name == "backend"
    assert (BACKEND_ROOT / "pyproject.toml").is_file()


def test_data_paths_are_absolute_and_nested_under_backend() -> None:
    assert constants.DATA_DIR.is_absolute()

    for path in (
        constants.RAW_DIR,
        constants.UPLOAD_DIR,
        constants.CACHE_DIR,
        constants.PROCESSED_DIR,
        constants.DB_PATH,
    ):
        assert constants.DATA_DIR in path.parents


def test_settings_does_not_redefine_data_dir() -> None:
    """
    Regression test.

    ``settings.data_dir`` once duplicated ``constants.DATA_DIR``, giving two
    competing sources of truth for the same path. Environment-varying values
    belong in settings; fixed project structure belongs in constants.
    """

    assert not hasattr(Settings(), "data_dir")

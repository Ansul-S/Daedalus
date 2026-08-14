"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from daedalus.api.main import app
from daedalus.config import constants

# The constants naming every directory the application writes to. Startup
# creates all of them and the database lives inside one, so a test that
# boots the app would otherwise write into the developer's real data/.
_WRITABLE_PATHS = ("DATA_DIR", "RAW_DIR", "UPLOAD_DIR", "CACHE_DIR", "PROCESSED_DIR")


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect every application write to a temporary directory."""

    for name in _WRITABLE_PATHS:
        monkeypatch.setattr(constants, name, tmp_path / name.lower())

    monkeypatch.setattr(constants, "DB_PATH", tmp_path / "daedalus.db")

    return tmp_path


@pytest.fixture
def client(data_dir: Path) -> Iterator[TestClient]:
    """
    A test client that runs the application lifespan.

    Using TestClient as a context manager is what triggers startup and
    shutdown — without it, lifespan code never executes.
    """

    with TestClient(app) as test_client:
        yield test_client

"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from daedalus.api.main import app


@pytest.fixture
def client() -> Iterator[TestClient]:
    """
    A test client that runs the application lifespan.

    Using TestClient as a context manager is what triggers startup and
    shutdown — without it, lifespan code never executes.
    """

    with TestClient(app) as test_client:
        yield test_client

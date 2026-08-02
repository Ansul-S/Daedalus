"""Tests for the application's top-level routes."""

from __future__ import annotations

from fastapi.testclient import TestClient

from daedalus.api.main import app
from daedalus.config import constants


def test_root_identifies_the_service(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert constants.APP_NAME in response.json()["message"]


def test_health_reports_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_every_route_path_is_unique() -> None:
    """
    Regression test.

    ``root`` and ``health`` were once both registered at "/", which made
    ``health`` permanently unreachable — FastAPI matches routes in
    registration order, so the first one always wins. A duplicate path is
    silently accepted at import time, so only a test catches it.
    """

    paths = [route.path for route in app.routes]  # type: ignore[attr-defined]

    assert len(paths) == len(set(paths)), f"duplicate route paths: {paths}"

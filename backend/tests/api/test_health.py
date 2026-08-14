"""Tests for the application's top-level routes."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.routing import BaseRoute

from daedalus.api.main import app
from daedalus.config import constants


def _api_routes(routes: Sequence[BaseRoute]) -> Iterator[APIRoute]:
    """Every registered route, including those inside included routers.

    ``include_router`` nests a router object in ``app.routes`` rather than
    flattening its routes into it, so reading ``app.routes`` directly sees
    only the routes declared on the app itself.
    """

    for route in routes:
        if isinstance(route, APIRoute):
            yield route

        included = getattr(route, "original_router", None)

        if included is not None:
            yield from _api_routes(included.routes)


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

    registered = [
        (route.path, method) for route in _api_routes(app.routes) for method in route.methods
    ]

    assert len(registered) == len(set(registered)), f"duplicate routes: {registered}"

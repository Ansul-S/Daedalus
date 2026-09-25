import json

from fastapi.routing import APIRoute

from app.api import documents, grading, health, practice, questions, ratings, search
from app.main import app
from scripts.openapi import write_schema

ROUTERS = [health, documents, search, questions, grading, practice, ratings]


def operations() -> list[dict]:
    return [op for ops in app.openapi()["paths"].values() for op in ops.values()]


def test_each_operation_is_named_after_its_handler() -> None:
    paths = app.openapi()["paths"]
    routes = [route for module in ROUTERS for route in module.router.routes]

    assert routes and all(isinstance(route, APIRoute) for route in routes)
    for route in routes:
        for method in route.methods:
            operation = paths[route.path][method.lower()]
            assert operation["operationId"] == route.endpoint.__name__, route.path
    assert sum(len(route.methods) for route in routes) == len(operations())


def test_operation_ids_are_unique() -> None:
    ids = [op["operationId"] for op in operations()]

    assert len(ids) == len(set(ids))
    assert "practice_next" in ids


def test_the_script_writes_the_whole_schema(tmp_path) -> None:
    path = tmp_path / "openapi.json"

    write_schema(path)

    written = json.loads(path.read_text("utf-8"))
    assert written == app.openapi()
    assert "/practice/next" in written["paths"]

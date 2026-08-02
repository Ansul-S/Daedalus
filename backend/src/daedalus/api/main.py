"""
FastAPI application entry point.

Defines the ASGI app and its top-level routes. All startup and shutdown
work is delegated to ``daedalus.core.lifespan``, so importing this module
has no side effects.

Run with:

    uv run uvicorn daedalus.api.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI

from daedalus.config import constants
from daedalus.core.lifespan import lifespan

__all__ = ["app"]


app = FastAPI(
    title=constants.APP_NAME,
    version=constants.VERSION,
    lifespan=lifespan,
)


# Routes


@app.get("/")
def root() -> dict[str, str]:
    """Identify the service."""

    return {"message": f"{constants.APP_NAME} is running"}


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""

    return {"status": "ok"}

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

from daedalus.api import answer, documents, search
from daedalus.config import constants
from daedalus.core.lifespan import lifespan

__all__ = ["app"]


app = FastAPI(
    title=constants.APP_NAME,
    version=constants.VERSION,
    lifespan=lifespan,
)

app.include_router(documents.router)
app.include_router(search.router)
app.include_router(answer.router)


# Routes


@app.get("/")
def root() -> dict[str, str]:
    """Identify the service."""

    return {"message": f"{constants.APP_NAME} is running"}


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""

    return {"status": "ok"}

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute

from app.api import (
    account,
    documents,
    grading,
    health,
    interviews,
    practice,
    questions,
    ratings,
    search,
)
from app.core.config import get_settings
from app.grading.limits import LimitReached
from app.llm.models import embedding_model
from app.llm.tracing import tracing


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    with tracing(settings, "api"):
        # Query embeddings need Ollama, which only runs locally; production search is
        # keyword-only.
        if settings.environment == "local":
            async with embedding_model(settings) as embedder:
                app.state.embedder = embedder
                yield
        else:
            app.state.embedder = None
            yield


def operation_id(route: APIRoute) -> str:
    """Name each operation after its handler: the typed frontend client generated from the
    schema then calls `practiceNext()` rather than `practiceNextPracticeNextGet()`."""
    return route.name


app = FastAPI(
    title="Daedalus API",
    version="0.1.0",
    lifespan=lifespan,
    generate_unique_id_function=operation_id,
    root_path=get_settings().root_path,
    # FastAPI's own OpenTelemetry would record every request, and validation failures with the
    # body sent and errors with their stack traces, into the global providers, and send them
    # wherever an OTEL_* variable points. Traces go through app.llm.tracing alone, without
    # what was asked or answered.
    telemetry={"tracing": False, "metrics": False, "logs": False, "auto_configure": False},
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_exception_handler(LimitReached, grading.limit_reached)
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(questions.router)
app.include_router(grading.router)
app.include_router(practice.router)
app.include_router(ratings.router)
app.include_router(interviews.router)
app.include_router(account.router)

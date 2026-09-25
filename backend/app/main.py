from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute

from app.api import documents, grading, health, practice, questions, ratings, search
from app.core.config import get_settings
from app.llm.embeddings import Embedder


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    # Query embeddings need Ollama, which only runs locally; production search is keyword-only.
    if settings.environment == "local":
        async with Embedder(settings) as embedder:
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
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(questions.router)
app.include_router(grading.router)
app.include_router(practice.router)
app.include_router(ratings.router)

"""Dependency checks shared by `GET /health/deps` and `scripts/check_setup.py`, and one
only the API can make: what its own tracing has sent."""

from pathlib import Path
from typing import Literal

import httpx
from alembic.script import ScriptDirectory
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import Settings
from app.llm.tracing import Tracing, switched_off

Status = Literal["ok", "warn", "fail"]
MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "db" / "migrations"


class Check(BaseModel):
    name: str
    status: Status
    detail: str


async def check_database(engine: AsyncEngine) -> Check:
    try:
        async with engine.connect() as conn:
            version = await conn.scalar(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            )
    except Exception as exc:  # any failure here means "not usable"; say which kind
        return Check(
            name="postgres",
            status="fail",
            detail=f"cannot connect ({type(exc).__name__}); run `make db-up`",
        )
    if version is None:
        return Check(name="postgres", status="fail", detail="connected, but pgvector is missing")
    return Check(name="postgres", status="ok", detail=f"connected, pgvector {version}")


async def check_schema(engine: AsyncEngine) -> Check:
    latest = ScriptDirectory(str(MIGRATIONS_DIR)).get_current_head()
    try:
        async with engine.connect() as conn:
            current = await conn.scalar(text("SELECT version_num FROM alembic_version"))
    except DBAPIError:  # the version table does not exist yet
        current = None
    if current == latest:
        return Check(name="database schema", status="ok", detail=f"up to date ({latest})")
    found = f"at {current}" if current else "not created"
    return Check(
        name="database schema",
        status="fail",
        detail=f"{found}, latest is {latest}; run `make migrate`",
    )


def is_installed(model: str, installed: set[str]) -> bool:
    # Ollama reports untagged models as "<name>:latest".
    return model in installed or f"{model}:latest" in installed


async def check_ollama(settings: Settings) -> list[Check]:
    try:
        async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=3) as client:
            response = await client.get("/api/tags")
            response.raise_for_status()
    except httpx.HTTPError as exc:
        return [
            Check(
                name="ollama",
                status="fail",
                detail=f"not reachable at {settings.ollama_base_url} ({type(exc).__name__}); "
                "run `make ollama`",
            )
        ]

    installed = {model["name"] for model in response.json().get("models", [])}
    checks = [Check(name="ollama", status="ok", detail=f"running, {len(installed)} models")]
    for model in [*settings.local_chat_models, settings.embedding_model]:
        if is_installed(model, installed):
            checks.append(Check(name=f"model {model}", status="ok", detail="installed"))
        else:
            checks.append(
                Check(
                    name=f"model {model}",
                    status="fail",
                    detail=f"missing; run `ollama pull {model}`",
                )
            )
    return checks


def check_cloud_keys(settings: Settings) -> list[Check]:
    keys = {"groq": settings.groq_api_key, "gemini": settings.gemini_api_key}
    # Locally, cloud keys are optional. In production they are the only models available,
    # so at least one must be set.
    any_key = any(key is not None for key in keys.values())
    missing: Status = "warn" if settings.environment == "local" or any_key else "fail"
    return [
        Check(name=f"{name} API key", status="ok", detail="set")
        if key is not None
        else Check(
            name=f"{name} API key",
            status=missing,
            detail=f"not set (add {name.upper()}_API_KEY to .env)",
        )
        for name, key in keys.items()
    ]


def check_tracing(settings: Settings) -> Check:
    """Whether model calls are traced, and what a trace holds. Off is a choice, not a
    problem, unless only one of the two keys is set."""
    off = switched_off(settings)
    if off is None:
        holds = (
            "everything, prompts and answers included (LANGFUSE_CONTENT)"
            if settings.langfuse_content
            else "timings, tokens and models, no prompts or answers"
        )
        detail = f"to Langfuse at {settings.langfuse_base_url}: {holds}"
        return Check(name="tracing", status="ok", detail=detail)
    one_key = (settings.langfuse_public_key is None) != (settings.langfuse_secret_key is None)
    return Check(name="tracing", status="warn" if one_key else "ok", detail=f"off: {off}")


def check_traces_sent(started: Tracing | None) -> Check:
    """What the running API has sent to Langfuse, while tracing is on: from inside the process
    that traces, as the API's own health route sees it."""
    if started is None:
        return Check(name="traces sent", status="warn", detail="tracing was not started here")
    sent, refused = started.sender.results["sent"], started.sender.results["refused"]
    batches = f"{sent} batch{'' if sent == 1 else 'es'} taken, {refused} refused"
    return Check(
        name="traces sent", status="warn" if refused else "ok", detail=f"since start: {batches}"
    )


def check_limits(settings: Settings) -> Check:
    """The daily limits on grading. None is a choice locally; a limit of 0 stops grading."""
    per_user, in_all = settings.daily_grades_per_user, settings.daily_grades
    if per_user is None and in_all is None:
        return Check(name="daily limits", status="ok", detail="none: grading is not limited")
    parts = [
        f"{per_user} grade{'' if per_user == 1 else 's'} a user a practice day"
        if per_user is not None
        else None,
        f"{in_all} in all over 24 hours" if in_all is not None else None,
    ]
    detail = ", ".join(part for part in parts if part)
    if 0 in (per_user, in_all):
        return Check(name="daily limits", status="warn", detail=f"{detail}: grading is off")
    return Check(name="daily limits", status="ok", detail=detail)


async def run_checks(settings: Settings, engine: AsyncEngine) -> list[Check]:
    database = await check_database(engine)
    checks = [database]
    if database.status == "ok":
        checks.append(await check_schema(engine))
    if settings.fake_models:
        # Neither Ollama nor a cloud key is used, and every grade is made up: say so instead.
        detail = "stand-ins for testing (FAKE_MODELS): nothing is sent to a model"
        models = Check(name="models", status="warn", detail=detail)
        return [*checks, models, check_tracing(settings), check_limits(settings)]
    if settings.environment == "local":
        checks += await check_ollama(settings)
    checks += check_cloud_keys(settings)
    checks.append(check_tracing(settings))
    checks.append(check_limits(settings))
    return checks

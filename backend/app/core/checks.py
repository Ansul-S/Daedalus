"""Dependency checks shared by `GET /health/deps` and `scripts/check_setup.py`."""

from typing import Literal

import httpx
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import Settings

Status = Literal["ok", "warn", "fail"]


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


async def run_checks(settings: Settings, engine: AsyncEngine) -> list[Check]:
    checks = [await check_database(engine)]
    if settings.environment == "local":
        checks += await check_ollama(settings)
    checks += check_cloud_keys(settings)
    return checks

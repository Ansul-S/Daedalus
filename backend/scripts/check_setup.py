"""Check what Daedalus needs: the database, Ollama and its models, and cloud API keys.

Run from backend/:  uv run python -m scripts.check_setup [--live]
"""

import argparse
import asyncio
import sys
import time

import httpx
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models import Model

from app.core.checks import Check, run_checks
from app.core.config import Settings, get_settings
from app.db.session import engine
from app.llm import models

ICONS = {"ok": "✓", "warn": "!", "fail": "✗"}

# The provider accepted the key but is rate-limiting or overloaded; free tiers do this often.
BUSY_STATUS_CODES = {429, 503}


def show(checks: list[Check]) -> None:
    for check in checks:
        print(f"  {ICONS[check.status]} {check.name:<34} {check.detail}")


async def ping_chat_model(model: Model) -> Check:
    name = f"reply from {model.system}:{model.model_name}"
    start = time.perf_counter()
    try:
        result = await Agent(model).run("Reply with exactly one word: ok")
    except ModelHTTPError as exc:
        if exc.status_code in BUSY_STATUS_CODES:
            detail = f"HTTP {exc.status_code}: provider busy or rate-limited, try again later"
            return Check(name=name, status="warn", detail=detail)
        return Check(name=name, status="fail", detail=f"HTTP {exc.status_code}: {exc}"[:200])
    except Exception as exc:  # report the failure and keep checking the other models
        return Check(name=name, status="fail", detail=f"{type(exc).__name__}: {exc}"[:200])
    elapsed = time.perf_counter() - start
    return Check(name=name, status="ok", detail=f"{result.output.strip()[:30]!r} in {elapsed:.1f}s")


async def ping_embedding_model(settings: Settings) -> Check:
    name = f"vector from ollama:{settings.embedding_model}"
    try:
        async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=120) as client:
            response = await client.post(
                "/api/embed", json={"model": settings.embedding_model, "input": "ok"}
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        return Check(name=name, status="fail", detail=f"{type(exc).__name__}: {exc}"[:200])
    dimensions = len(response.json()["embeddings"][0])
    return Check(name=name, status="ok", detail=f"{dimensions} dimensions")


async def main(live: bool) -> int:
    settings = get_settings()
    checks = await run_checks(settings, engine)
    await engine.dispose()
    print("Setup checks:")
    show(checks)

    if live:
        print("\nTest prompts (the first call to a local model includes loading it):")
        installed = {
            check.name.removeprefix("model ")
            for check in checks
            if check.name.startswith("model ") and check.status == "ok"
        }
        chat_models = [
            models.ollama(settings, name)
            for name in settings.local_chat_models
            if name in installed
        ]
        chat_models += [models.groq(settings), models.gemini(settings)]
        for model in chat_models:
            if model is not None:
                result = await ping_chat_model(model)
                show([result])
                checks.append(result)
        if settings.embedding_model in installed:
            result = await ping_embedding_model(settings)
            show([result])
            checks.append(result)

    failed = sum(check.status == "fail" for check in checks)
    print(f"\n{failed} problem(s) found." if failed else "\nAll required checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the Daedalus setup.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="also send a one-word test prompt to each model (uses a little free-tier quota)",
    )
    sys.exit(asyncio.run(main(parser.parse_args().live)))

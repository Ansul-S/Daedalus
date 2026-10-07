from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.checks import Check, check_traces_sent, run_checks
from app.core.config import Settings, get_settings
from app.db.session import engine
from app.llm import tracing

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """The API process is up."""
    return {"status": "ok"}


@router.get("/health/deps")
async def dependencies(settings: Annotated[Settings, Depends(get_settings)]) -> list[Check]:
    """Database, local models and cloud API keys (whether keys are set, never their values),
    and, while tracing is on, what this process has sent."""
    checks = await run_checks(settings, engine)
    if tracing.switched_off(settings) is None:
        checks.append(check_traces_sent(tracing.current()))
    return checks

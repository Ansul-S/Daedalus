from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.checks import Check, run_checks
from app.core.config import Settings, get_settings
from app.db.session import engine

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """The API process is up."""
    return {"status": "ok"}


@router.get("/health/deps")
async def dependencies(settings: Annotated[Settings, Depends(get_settings)]) -> list[Check]:
    """Database, local models and cloud API keys (whether keys are set, never their values)."""
    return await run_checks(settings, engine)

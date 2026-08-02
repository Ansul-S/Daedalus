"""
Application startup and shutdown.

Everything that must happen exactly once when the API process starts, and
once when it stops, belongs here. FastAPI runs this via its ``lifespan``
argument, which guarantees startup completes before the first request is
served and shutdown runs after the last one finishes.

Keeping this out of the route module means importing the app never triggers
side effects — which is what makes the application testable.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from daedalus.config import constants
from daedalus.core.logging import configure_logging

__all__ = ["lifespan"]


logger = logging.getLogger(__name__)


# Startup Tasks


def _create_data_directories() -> None:
    """
    Create the directories the application writes to.

    Safe to call repeatedly. Doing this at startup means the ingestion
    pipeline never has to defensively check whether its target exists.
    """

    for directory in (
        constants.DATA_DIR,
        constants.RAW_DIR,
        constants.UPLOAD_DIR,
        constants.CACHE_DIR,
        constants.PROCESSED_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


# Lifespan


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Manage the application lifecycle.

    Code before ``yield`` runs at startup, code after it runs at shutdown.
    Logging is configured first so that every subsequent startup step is
    itself logged.
    """

    configure_logging()

    logger.info("Starting %s v%s", constants.APP_NAME, constants.VERSION)

    _create_data_directories()

    yield

    logger.info("Shutting down %s", constants.APP_NAME)

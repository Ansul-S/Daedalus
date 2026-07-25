"""
Central logging configuration for Daedalus.

This module configures the application's root logger exactly once during
startup. All other modules should obtain loggers using:

    import logging

    logger = logging.getLogger(__name__)
"""

from __future__ import annotations

import logging
import sys

from daedalus.config import settings

__all__ = ["configure_logging"]


# Logging Format

LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)-8s | "
    "%(name)s | "
    "%(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"



# Logging Configuration

def configure_logging() -> None:
    """
    Configure the application's root logger.

    This function should be called exactly once during application startup,
    before any modules emit log messages.

    Log level is read from the application's runtime settings.
    """

    # Convert string log level (e.g. "INFO", "DEBUG") to logging constant.
    log_level = getattr(
        logging,
        settings.log_level.upper(),
        logging.INFO,
    )

    handlers = [
        logging.StreamHandler(sys.stdout),
    ]

    logging.basicConfig(
        level=log_level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=handlers,
        force=True,  # Replace existing handlers (e.g. uvicorn defaults)
    )

    # Reduce noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    logging.getLogger().info("Logging initialized.")
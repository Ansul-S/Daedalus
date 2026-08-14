"""
Application runtime configuration
loaded from env variables and .env using pydantic-settings.

Everything here varies by environment. Values that are fixed properties of
the project belong in ``constants.py`` instead.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# The backend/ directory, resolved from this file's location rather than the
# process working directory — so `.env` is found no matter where uvicorn is
# launched from.
BACKEND_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Runtime configuration."""

    # Application

    app_name: str = "Daedalus"
    debug: bool = False

    host: str = "127.0.0.1"
    port: int = 8000

    # Models

    llm_model: str = "qwen3:8b"
    routing_model: str = "qwen3:4b"
    vision_model: str = "qwen2.5vl:7b"
    embedding_model: str = "BAAI/bge-m3"

    # Which embedder the API serves with. "fake" is deterministic and needs
    # no model, which is what lets the test suite and a fresh clone exercise
    # the full pipeline; it produces meaningless neighbourhoods, so it is
    # never the default.
    embedding_backend: str = "bge"

    # The generation counterpart. "fake" returns scripted text and contacts
    # nothing, so the API can be exercised without Ollama running.
    llm_backend: str = "ollama"

    # Greedy. Grounded answers should restate the sources rather than
    # improvise, and a benchmark cannot be built on a generator that
    # answers differently each run — at 0.1 the same question over the
    # same context alternated between answering and refusing.
    llm_temperature: float = 0.0

    # Generous: a local 8B model on CPU can take a minute for a long answer.
    llm_timeout_seconds: float = 120.0

    # Uploads

    # Ingestion reads the whole file, so an unbounded upload is an
    # unbounded memory and disk cost.
    max_upload_mb: int = 100

    # Services

    ollama_url: str = "http://localhost:11434"

    # Logging

    log_level: str = "INFO"

    # Pydantic settings

    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_prefix="DAEDALUS_",
        case_sensitive=False,
    )


settings = Settings()

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Read from environment variables first, then the repo-root `.env` file."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    # "production" means a free cloud host: no Ollama, cloud models only.
    environment: Literal["local", "production"] = "local"
    cors_origins: list[str] = ["http://localhost:3000"]

    database_url: str = "postgresql+psycopg://daedalus:daedalus@localhost:5433/daedalus"

    # Local models served by Ollama
    ollama_base_url: str = "http://localhost:11434"
    grader_model: str = "qwen3.5:9b"
    helper_model: str = "qwen3.5:4b"
    second_opinion_model: str = "gemma4:12b"
    embedding_model: str = "qwen3-embedding:0.6b"

    # Kept small and constant: a different value on each call makes Ollama reload the model.
    embedding_num_ctx: int = 2048

    # Free cloud models (optional locally)
    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-120b"
    gemini_api_key: SecretStr | None = None
    # Pinned rather than the "-latest" alias: the alias moved to 3.8 Flash, which answered
    # 2 of 13 trial requests and whose profile drops the thinking setting.
    gemini_model: str = "gemini-3.5-flash"

    # Ingestion (local only). Uploads and downloaded papers are stored under data_dir.
    data_dir: Path = REPO_ROOT / "data"
    max_upload_mb: int = 50
    # Chunks grow until the next block would pass the maximum; a new section or subsection
    # starts a new chunk once the current one has the minimum.
    chunk_min_tokens: int = 300
    chunk_max_tokens: int = 800
    # Hugging Face tokenizer of the embedding model, used to measure chunk sizes.
    tokenizer_model: str = "Qwen/Qwen3-Embedding-0.6B"

    # Topic map: how close two concept tags have to be, in cosine similarity, to become one
    # topic. Higher keeps topics narrow, lower merges more of them.
    topic_similarity: float = 0.8
    # Above this similarity two questions are the same question in other words. Measured on
    # seven pairs: real near-duplicates scored 0.75 to 0.86, while two different questions
    # about one topic reached only 0.61, so the gap to sit in is narrow and low. The
    # embedding model keeps short texts much closer together than whole passages.
    duplicate_similarity: float = 0.7

    @property
    def local_chat_models(self) -> list[str]:
        return [self.grader_model, self.helper_model, self.second_opinion_model]


@lru_cache
def get_settings() -> Settings:
    return Settings()

"""
Application runtime configuration
loaded from env variables and .env using pydantic-settings.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Runtime configuration."""

    #Application

    app_name: str = "Daedalus"
    debug: bool = False

    host: str = "127.0.0.1"
    port: int = 8000

    #Models

    llm_model: str = "qwen3:8b"
    routing_model: str = "qwen3:4b"
    vision_model: str = "qwen2.5vl:7b"
    embedding_model: str = "BAAI/bge-m3"

    #Services

    ollama_url: str = "http://localhost:11434"

    #Project paths

    data_dir: Path = Path("data")

    #Pydantic settings

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DAEDALUS_",
        case_sensitive=False,
    )

settings = Settings()
"""
Project-wide immutable constants.

Do Not store environment-specific values here.
"""

from pathlib import Path

#Application

APP_NAME = "Daedalus"
VERSION = "0.1.0"

#Directory structure

DATA_DIR = Path("data")

RAW_DIR = DATA_DIR / "raw"
UPLOAD_DIR = DATA_DIR / "uploads"
CACHE_DIR = DATA_DIR / "cache"
PROCESSED_DIR = DATA_DIR / "processed"

DB_PATH = DATA_DIR / "daedalus.db"

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".md",
    ".ipynb",
}

#Document Processing

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

#Retrieval

DEFAULT_TOP_K = 5

HYBRID_CANDIDATES = 40
RERANK_TOP_K = 4
RRF_K = 60
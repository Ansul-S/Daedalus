"""
Project-wide immutable constants.

Do Not store environment-specific values here.
"""

from pathlib import Path

# Application

APP_NAME = "Daedalus"
VERSION = "0.1.0"

# Directory structure

# Anchored to backend/ rather than the process working directory, so data
# always lands in the same place regardless of where the app is launched.
BACKEND_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = BACKEND_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
UPLOAD_DIR = DATA_DIR / "uploads"
CACHE_DIR = DATA_DIR / "cache"
PROCESSED_DIR = DATA_DIR / "processed"

DB_PATH = DATA_DIR / "daedalus.db"

# Evaluation

# Committed, unlike data/. Frozen parsed text and the labelled datasets are
# the fixed reference every reported metric is measured against — see
# docs/pipelines/EVALUATION_ENGINE.md. Re-parsing is a versioned migration,
# not a side effect of running the pipeline.
CORPUS_DIR = BACKEND_ROOT.parent / "corpus"

EVAL_DIR = BACKEND_ROOT / "eval"
EVAL_PARSED_DIR = EVAL_DIR / "parsed"
EVAL_DATASETS_DIR = EVAL_DIR / "datasets"

EVAL_MANIFEST_PATH = EVAL_PARSED_DIR / "manifest.json"

# Query stratification for the retrieval dataset. Each type stresses a
# different half of hybrid retrieval, which is what makes the benchmark
# discriminative rather than a single averaged number.
QUERY_TYPES = {
    "exact_term",
    "conceptual",
    "definitional",
    "multi_hop",
    "comparative",
    "unanswerable",
}

# Tune freely against dev; touch test only when reporting final numbers.
SPLITS = {"dev", "test"}

# 2 = essential, the question cannot be answered without this span.
# 1 = supporting.
GRADE_ESSENTIAL = 2
GRADE_SUPPORTING = 1

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".md",
    ".ipynb",
    ".png",
    ".jpg",
    ".jpeg",
}

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
}

# Document Processing

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# How a span of text was extracted from its source document.
# Recorded per chunk so retrieval metrics can be sliced by ingestion path.
EXTRACTION_TEXT = "text"
EXTRACTION_OCR = "ocr"
EXTRACTION_VISION = "vision"
EXTRACTION_NOTEBOOK = "notebook"

EXTRACTION_METHODS = {
    EXTRACTION_TEXT,
    EXTRACTION_OCR,
    EXTRACTION_VISION,
    EXTRACTION_NOTEBOOK,
}

# A PDF page yielding fewer than this many characters of text layer is
# treated as scanned and routed to OCR. Pages with a real text layer in the
# evaluation corpus produce 1,770-4,006 characters; scanned pages produce
# essentially none. See ADR-009.
TEXT_LAYER_MIN_CHARS = 100

# Ingestion Status

# The lifecycle of a document record. Mirrors the CHECK constraint on
# documents.status — change one and the other rejects every write.
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

DOCUMENT_STATUSES = {
    STATUS_PENDING,
    STATUS_PROCESSING,
    STATUS_COMPLETED,
    STATUS_FAILED,
}

# Embeddings

# BAAI/bge-m3 produces 1024-dimensional vectors. Fixed at table creation
# time by the vec0 virtual table, so changing it requires a re-index.
EMBEDDING_DIM = 1024

# Texts encoded per forward pass. Large enough to keep the model busy, small
# enough that a batch of long chunks fits in memory on a laptop GPU.
EMBEDDING_BATCH_SIZE = 32

# Retrieval

DEFAULT_TOP_K = 5

HYBRID_CANDIDATES = 40
RERANK_TOP_K = 4
RRF_K = 60

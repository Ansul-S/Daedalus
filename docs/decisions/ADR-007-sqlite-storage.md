# ADR-007: SQLite with sqlite-vec and FTS5 for Storage and Hybrid Retrieval

**Status:** Accepted
**Date:** 2026-08-02

---

## Context

Daedalus retrieves chunks of study material to ground question generation and answer evaluation. Retrieval is hybrid: a dense semantic search over BGE-M3 embeddings, combined with a lexical keyword search, fused by Reciprocal Rank Fusion.

Hybrid retrieval is not optional for this domain. Study material for AI/ML is dense with exact terms — `softmax`, `BM25`, `LoRA`, `RMSNorm` — where lexical matching outperforms embeddings. It is also full of conceptual paraphrase, where embeddings outperform keywords. Each half covers the other's failure mode.

This requires three capabilities: durable storage of chunks and metadata, an approximate nearest-neighbour index, and a full-text index. The deployment target is a single machine — a laptop for development, one small VM at most for a demo. There is no ops team and no budget for managed infrastructure.

## Decision

Use a single SQLite database file for all three.

- **Chunks and metadata** — ordinary SQLite tables
- **Vector search** — `sqlite-vec`, a loadable extension providing `vec0` virtual tables with KNN search
- **Lexical search** — SQLite's built-in `FTS5` module, which supplies BM25 ranking
- **Fusion** — Reciprocal Rank Fusion in application code, `k = 60` (already fixed in `config/constants.py`)

All of it lives in `data/daedalus.db`, the path already defined at `constants.py:23`.

Total new dependencies: **one** (`sqlite-vec`). FTS5 ships inside SQLite; `BackgroundTasks` ships inside FastAPI.

## Alternatives Considered

**Chroma** — the easiest API of the options and a genuinely good developer experience. Rejected because it solves only the dense half. Lexical search would still need FTS5 or `rank_bm25` alongside it, leaving two stores to keep consistent, and consistency between them becomes a real bug surface during re-indexing.

**Qdrant** — the most production-credible choice, with native hybrid search and proper filtering. Rejected because it requires running a separate service. Docker Compose for a demo raises the barrier for a reviewer who wants to clone the repo and run it, and the scale that justifies Qdrant is far beyond this project's corpus of a few thousand chunks.

**LanceDB** — embedded and columnar, strong on larger-than-memory datasets. Rejected as an unnecessary second storage engine when relational metadata is needed anyway.

**FAISS + separate metadata store** — fastest raw search, but it is an index, not a database. Persistence, deletion, and the mapping from vector back to chunk all become application code. Too much undifferentiated work.

## Consequences

**Positive**

- One file is the entire database. Backup is a file copy; resetting the index is a delete.
- Nothing to install or run beyond the Python environment. `git clone` then `uv sync` then run.
- Transactional consistency between chunks, embeddings, and the lexical index, because they share one connection and one transaction.
- Metadata filtering (by document, page, or topic) is plain SQL rather than a vendor-specific filter DSL.

**Negative**

- `sqlite-vec` performs brute-force KNN — it scans every vector rather than using an ANN index. Linear in corpus size. Acceptable at the thousands-of-chunks scale expected here; it would not be at millions.
- SQLite handles one writer at a time. Concurrent ingestion of multiple documents must be serialised. WAL mode should be enabled so reads are not blocked by the writer.
- `sqlite-vec` is a loadable extension, so it needs a Python built with `enable_load_extension` support. Verified working on the project's uv-managed Python 3.13 with SQLite 3.53.1; macOS system Python does **not** support this and will fail.
- Migrating to Qdrant later means rewriting the storage adapter. This is contained: `interfaces/retrieval.py` defines the contract, and only the concrete implementation in `storage/` would change.

## Verification

A spike confirmed the full path before adoption: `vec0` KNN search, FTS5 BM25 ranking, and RRF fusion over both, in a single database, returning correctly ranked results.

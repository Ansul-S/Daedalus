# ADR-008: FastAPI BackgroundTasks for Document Ingestion

**Status:** Accepted
**Date:** 2026-08-02

---

## Context

Ingesting a document is slow. A scanned PDF may need OCR, then parsing, chunking, and embedding generation across hundreds of chunks. This runs from seconds to several minutes.

An HTTP upload request cannot wait for that. Browsers and proxies time out, and holding a connection open for minutes is poor design regardless. The upload endpoint must accept the file, return immediately, and let processing continue in the background while the client polls for status.

The `worker/` package was scaffolded early with no execution model chosen and no queueing dependency installed.

## Decision

Use FastAPI's built-in `BackgroundTasks` for ingestion.

The upload endpoint validates the file, writes it to disk, creates a document record with status `pending`, schedules the ingestion pipeline as a background task, and returns `202 Accepted` with a document ID. The pipeline advances the record through `processing` to `completed` or `failed`. A status endpoint lets the client poll.

Ingestion status lives in the same SQLite database as everything else (see ADR-007), so progress survives a restart even though the in-flight task does not.

## Alternatives Considered

**ARQ or RQ with Redis** — a real job queue with retries, scheduling, and a separate worker process. Rejected for now because it adds a service to run, a second process to deploy, and a dependency on Redis, in exchange for guarantees this project does not yet need. A single user uploading their own study material is not a workload that requires a broker.

**Celery** — the same reasoning, with more configuration surface.

**Synchronous processing in the request** — simplest to write, but the endpoint would block for minutes and time out. Not viable.

**`asyncio.create_task`** — roughly what `BackgroundTasks` does, but without FastAPI's integration into the response lifecycle and error handling. No advantage.

## Consequences

**Positive**

- Zero new dependencies and zero new processes.
- The background task runs in the same process, so it shares the loaded embedding model. A separate worker would need its own copy of BGE-M3 in memory, roughly 2 GB.
- Simple to reason about and simple to demonstrate.

**Negative**

- **Tasks do not survive a restart.** A crash mid-ingestion leaves a document stuck in `processing`. Mitigation: on startup, reset any `processing` records to `failed` so they can be retried, and make ingestion idempotent by keying on a content hash.
- **No automatic retries.** Failures must be re-triggered manually through the API.
- **CPU-bound work blocks the event loop.** Embedding generation and OCR are not async-friendly. The pipeline must run in a thread pool — defining the ingestion function with `def` rather than `async def` makes FastAPI do this automatically, or use `run_in_threadpool` explicitly.
- **No concurrency control.** Several simultaneous uploads could exhaust memory or contend for SQLite's single writer. A semaphore limiting concurrent ingestion to one or two is needed.

## Revisit If

Ingestion needs to survive restarts, multiple users upload concurrently, or the backend is deployed across more than one process. At that point the pipeline moves behind a real queue — the pipeline function itself does not change, only what invokes it.

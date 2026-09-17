"""Postgres as the ingestion queue.

The API and the CLI only record documents and queue jobs. A worker claims queued jobs one at
a time (`FOR UPDATE SKIP LOCKED`), so parsing never runs inside the API process. An advisory
lock lets only one process ingest at a time: two Docling instances next to the local models
would not fit in 16 GB.

This module imports nothing heavy, so the API can use it.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.db.models import Document, Job
from app.ingest.storage import StoredFile

# Arbitrary key for pg_try_advisory_lock
INGEST_LOCK_KEY = 7_203_117
ACTIVE_STATUSES = ("queued", "running")


@dataclass
class Enqueued:
    document: Document
    # None when the document is already ingested and no new job was needed
    job: Job | None
    message: str


async def add_file(
    session: AsyncSession,
    stored: StoredFile,
    filename: str,
    options: dict[str, Any],
    force: bool = False,
) -> Enqueued:
    document = await session.scalar(select(Document).where(Document.sha256 == stored.sha256))
    if document is None:
        document = Document(
            source_type=stored.source_type,
            # Replaced by the title found while parsing
            title=PurePath(filename).stem,
            filename=filename,
            path=stored.path,
            sha256=stored.sha256,
        )
        session.add(document)
        await session.flush()
    return await _enqueue(session, document, options, force)


async def add_arxiv(
    session: AsyncSession,
    arxiv_id: str,
    version: str | None,
    options: dict[str, Any],
    force: bool = False,
) -> Enqueued:
    document = await session.scalar(select(Document).where(Document.arxiv_id == arxiv_id))
    if document is None:
        document = Document(source_type="arxiv", title=f"arXiv:{arxiv_id}", arxiv_id=arxiv_id)
        session.add(document)
        await session.flush()
    # Asking for a different version than the stored one means ingesting again.
    force = force or (version is not None and version != document.arxiv_version)
    return await _enqueue(session, document, {**options, "version": version}, force)


async def _enqueue(
    session: AsyncSession, document: Document, options: dict[str, Any], force: bool
) -> Enqueued:
    active = await session.scalar(
        select(Job)
        .where(Job.document_id == document.id, Job.status.in_(ACTIVE_STATUSES))
        .order_by(Job.id.desc())
        .limit(1)
    )
    if active is not None:
        return Enqueued(document, active, f"already {active.status}")
    if document.status == "ready" and not force:
        return Enqueued(document, None, "already ingested (use force to ingest it again)")
    job = Job(document_id=document.id, options=options)
    session.add(job)
    await session.flush()
    return Enqueued(document, job, "queued")


async def claim_next_job(session: AsyncSession) -> int | None:
    """Mark the oldest queued job as running and return its id."""
    job_id = await session.scalar(
        text(
            """
            UPDATE jobs
            SET status = 'running', started_at = now(), finished_at = NULL,
                error = NULL, progress = 'starting', attempts = attempts + 1
            WHERE id = (
                SELECT id FROM jobs
                WHERE status = 'queued'
                ORDER BY created_at, id
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING id
            """
        )
    )
    await session.commit()
    return job_id


async def requeue_interrupted(session: AsyncSession) -> int:
    """Jobs still marked running belong to a process that stopped. Only safe to call while
    holding the ingest lock, since then no other process can be running them."""
    result = await session.execute(
        text(
            "UPDATE jobs SET status = 'queued', progress = 'queued again after an interruption' "
            "WHERE status = 'running'"
        )
    )
    await session.commit()
    return result.rowcount


@asynccontextmanager
async def ingest_lock(engine: AsyncEngine) -> AsyncIterator[bool]:
    """Yields whether this process may ingest. The lock lives as long as the connection."""
    async with engine.connect() as connection:
        acquired = bool(
            await connection.scalar(
                text("SELECT pg_try_advisory_lock(:key)"), {"key": INGEST_LOCK_KEY}
            )
        )
        try:
            yield acquired
        finally:
            if acquired:
                await connection.execute(
                    text("SELECT pg_advisory_unlock(:key)"), {"key": INGEST_LOCK_KEY}
                )

"""
The ingestion pipeline, end to end.

Ties together what the earlier stages built: parse, chunk, embed, store.
This is the function the API hands to ``BackgroundTasks``, so it owns two
responsibilities the individual stages do not.

**It never raises.** A background task has nobody to report to — an
exception escaping here is logged by Starlette and then lost, leaving the
document stuck in ``processing`` forever. Every failure is instead
recorded on the document row, where the status endpoint can report it.

**It opens its own connection.** The request that scheduled this work has
already returned by the time it runs, and the connection that served that
request is closed. Per ADR-008 the function is also deliberately ``def``
rather than ``async def``, so FastAPI runs it in a thread pool: embedding
is CPU-bound and would otherwise block the event loop for every other
request in the process.
"""

from __future__ import annotations

import logging
from pathlib import Path

from daedalus.config import constants
from daedalus.db import get_connection
from daedalus.ingestion.chunker import chunk_text
from daedalus.ingestion.router import parse
from daedalus.interfaces.embedding import Embedder
from daedalus.storage import chunks as chunk_store
from daedalus.storage import documents

__all__ = ["ingest_document", "source_type_for"]


logger = logging.getLogger(__name__)


_SOURCE_TYPES = {
    ".pdf": "pdf",
    ".md": "markdown",
    ".ipynb": "notebook",
}


def source_type_for(path: Path) -> str:
    """
    Classify a file by extension.

    Coarser than the evaluation corpus's own labels (``arxiv``,
    ``course_notes``), which encode where a document came from — something
    an upload cannot know.
    """

    suffix = path.suffix.lower()

    if suffix in constants.IMAGE_EXTENSIONS:
        return "image"

    return _SOURCE_TYPES.get(suffix, "unknown")


def ingest_document(path: Path, doc_id: str, embedder: Embedder) -> None:
    """
    Parse, chunk, embed, and index one document.

    Assumes the document row already exists in ``pending`` — the upload
    endpoint creates it, so that a status query between the response and
    the first byte of work still finds something to report.
    """

    logger.info("Ingesting %s as %s", path.name, doc_id)

    try:
        with get_connection() as connection:
            documents.mark_processing(connection, doc_id)

            parsed = parse(path, source_type=source_type_for(path), doc_id=doc_id)
            chunks = chunk_text(parsed.text, parsed.segments)

            # Embedding is the slow step. Doing it before opening the write
            # transaction keeps SQLite's single writer free while the model
            # runs, which matters because WAL only lets readers overlap a
            # writer, not a second writer.
            embeddings = embedder.embed_documents([chunk.text for chunk in chunks])

            chunk_store.replace(connection, doc_id, chunks, embeddings)

            documents.mark_completed(
                connection,
                doc_id,
                n_pages=parsed.n_pages,
                n_chunks=len(chunks),
            )

        logger.info("Ingested %s: %d chunks", doc_id, len(chunks))

    except Exception as error:
        # Broad by design: whatever went wrong, the document must not be
        # left claiming to be in progress.
        logger.exception("Ingestion failed for %s", doc_id)

        try:
            with get_connection() as connection:
                documents.mark_failed(connection, doc_id, f"{type(error).__name__}: {error}")
        except Exception:  # pragma: no cover - the database itself is gone
            logger.exception("Could not record the failure of %s", doc_id)

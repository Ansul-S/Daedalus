"""Runs ingestion jobs: parse the source, pack it into chunks, embed them, store them.

A document's chunks are replaced in one transaction, so search never sees a half-ingested
document, and a failed re-ingestion leaves the previous chunks in place.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, insert, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.models import Chunk, Document, Job
from app.ingest.arxiv import ArxivClient
from app.ingest.chunking import SECTION_SEPARATOR, ChunkDraft, ParsedDocument, pack_blocks
from app.ingest.queue import claim_next_job
from app.ingest.tokens import token_counter
from app.llm.embeddings import Embedder

log = logging.getLogger(__name__)

Reporter = Callable[[Document, str], None]


def embedding_input(title: str, chunk: ChunkDraft) -> str:
    """Documents are embedded with their title and the sections they cover, so a chunk that
    never names its topic can still be found by it."""
    heading = SECTION_SEPARATOR.join([title, chunk.section]) if chunk.section else title
    return f"{heading}\n\n{chunk.text}"


class Ingestor:
    def __init__(
        self,
        settings: Settings,
        sessions: async_sessionmaker[AsyncSession],
        embedder: Embedder,
        arxiv: ArxivClient,
        report: Reporter | None = None,
    ) -> None:
        self.settings = settings
        self.sessions = sessions
        self.embedder = embedder
        self.arxiv = arxiv
        self.report = report or (
            lambda document, message: log.info("%s: %s", document.title, message)
        )

    async def process_queue(self, *, stop_when_empty: bool, poll_seconds: float = 2.0) -> int:
        """Run queued jobs one after another. Returns how many ran."""
        processed = 0
        while True:
            async with self.sessions() as session:
                job_id = await claim_next_job(session)
            if job_id is None:
                if stop_when_empty:
                    return processed
                await asyncio.sleep(poll_seconds)
                continue
            await self.run(job_id)
            processed += 1

    async def run(self, job_id: int) -> bool:
        """Run one claimed job. Returns whether it succeeded; failures are recorded on the job."""
        async with self.sessions() as session:
            job = await session.get_one(Job, job_id)
            document = await session.get_one(Document, job.document_id)
        timings: dict[str, float] = {}
        try:
            started = time.perf_counter()
            await self._progress(job, document, "parsing")
            parsed, fields = await self._parse(document, job.options)
            timings["parse_seconds"] = time.perf_counter() - started

            chunks = await asyncio.to_thread(self._chunk, parsed)

            started = time.perf_counter()

            async def embedding_progress(done: int, total: int) -> None:
                await self._progress(job, document, f"embedding {done}/{total} chunks")

            vectors = await self.embedder.embed_documents(
                [embedding_input(parsed.title, chunk) for chunk in chunks], embedding_progress
            )
            timings["embed_seconds"] = time.perf_counter() - started

            details = {
                **parsed.details,
                **{name: round(seconds, 1) for name, seconds in timings.items()},
                "chunks": len(chunks),
                "tokens": sum(chunk.token_count for chunk in chunks),
            }
            await self._store(job, document, parsed, fields, chunks, vectors, details)
        except Exception as exc:
            log.exception("job %s failed", job_id)
            await self._fail(job, document, exc)
            return False
        self.report(document, f"done: {len(chunks)} chunks in {sum(timings.values()):.0f} s")
        return True

    def _chunk(self, parsed: ParsedDocument) -> list[ChunkDraft]:
        return pack_blocks(
            parsed.blocks,
            token_counter(self.settings.tokenizer_model),
            max_tokens=self.settings.chunk_max_tokens,
            min_tokens=self.settings.chunk_min_tokens,
        )

    async def _progress(self, job: Job, document: Document, message: str) -> None:
        self.report(document, message)
        async with self.sessions() as session:
            await session.execute(update(Job).where(Job.id == job.id).values(progress=message))
            await session.commit()

    async def _parse(
        self, document: Document, options: dict[str, Any]
    ) -> tuple[ParsedDocument, dict[str, Any]]:
        """Returns the parsed document and the document fields learned while parsing."""
        # Parsers load their libraries (Docling, PyTorch) on first use.
        ocr = bool(options.get("ocr", False))
        formulas = bool(options.get("formulas", True))
        if document.source_type == "pdf":
            from app.ingest.pdf import parse_pdf

            path = self.settings.data_dir / (document.path or "")
            fallback = Path(document.filename or path.name).stem
            parsed = await asyncio.to_thread(
                parse_pdf, path, fallback_title=fallback, ocr=ocr, formulas=formulas
            )
            return parsed, {"title": parsed.title}

        if document.source_type == "notebook":
            from app.ingest.notebook import parse_notebook

            path = self.settings.data_dir / (document.path or "")
            fallback = Path(document.filename or path.name).stem
            parsed = await asyncio.to_thread(parse_notebook, path, fallback_title=fallback)
            return parsed, {"title": parsed.title}

        return await self._parse_arxiv(document, options, ocr=ocr, formulas=formulas)

    async def _parse_arxiv(
        self, document: Document, options: dict[str, Any], *, ocr: bool, formulas: bool
    ) -> tuple[ParsedDocument, dict[str, Any]]:
        arxiv_id = document.arxiv_id or ""
        metadata = await self.arxiv.metadata(arxiv_id)
        version = options.get("version") or metadata.latest_version
        page = await self.arxiv.html(arxiv_id, version)
        if page is not None:
            from app.ingest.arxiv_html import parse_arxiv_html

            url, html = page
            parsed = await asyncio.to_thread(parse_arxiv_html, html, title=metadata.title, url=url)
        else:
            from app.ingest.pdf import parse_pdf

            self.report(document, "no HTML version; parsing the PDF")
            url, path = await self.arxiv.pdf(arxiv_id, version)
            parsed = await asyncio.to_thread(
                parse_pdf, path, fallback_title=metadata.title, ocr=ocr, formulas=formulas
            )
            parsed.title = metadata.title
        parsed.details.update(
            source=url,
            abstract=metadata.abstract,
            categories=metadata.categories,
            latest_version=metadata.latest_version,
        )
        fields = {
            "title": metadata.title,
            "authors": metadata.authors,
            "license": metadata.license,
            "arxiv_version": version,
            "url": url,
        }
        return parsed, fields

    async def _store(
        self,
        job: Job,
        document: Document,
        parsed: ParsedDocument,
        fields: dict[str, Any],
        chunks: list[ChunkDraft],
        vectors: list[list[float]],
        details: dict[str, Any],
    ) -> None:
        rows = [
            {
                "document_id": document.id,
                "position": position,
                "section": chunk.section,
                "content_types": chunk.content_types,
                "text": chunk.text,
                "token_count": chunk.token_count,
                "page_start": chunk.start if parsed.locator == "page" else None,
                "page_end": chunk.end if parsed.locator == "page" else None,
                "cell_start": chunk.start if parsed.locator == "cell" else None,
                "cell_end": chunk.end if parsed.locator == "cell" else None,
                "anchor": chunk.anchor,
                "embedding": vector,
            }
            for position, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        async with self.sessions() as session, session.begin():
            await session.execute(delete(Chunk).where(Chunk.document_id == document.id))
            if rows:
                await session.execute(insert(Chunk), rows)
            await session.execute(
                update(Document)
                .where(Document.id == document.id)
                .values(status="ready", error=None, details=details, **fields)
            )
            await session.execute(
                update(Job)
                .where(Job.id == job.id)
                .values(status="done", progress=f"{len(rows)} chunks", finished_at=func.now())
            )

    async def _fail(self, job: Job, document: Document, exc: Exception) -> None:
        message = f"{type(exc).__name__}: {exc}"[:2000]
        self.report(document, f"failed: {message}")
        async with self.sessions() as session, session.begin():
            await session.execute(
                update(Job)
                .where(Job.id == job.id)
                .values(status="failed", error=message, finished_at=func.now())
            )
            # A document that was already searchable keeps its previous chunks and status.
            await session.execute(
                update(Document)
                .where(Document.id == document.id, Document.status != "ready")
                .values(status="failed", error=message)
            )

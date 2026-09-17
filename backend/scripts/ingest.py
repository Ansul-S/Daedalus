"""Ingest PDFs and notebooks (files or folders) and arXiv papers.

Run from the repo root:  make ingest SRC="data/notes.pdf data/notebooks 1706.03762"
or from backend/:        uv run --group ingest python -m scripts.ingest SOURCE [SOURCE ...]

The jobs run in this process, unless a worker (`make worker`) is running: then the worker
takes them and this command waits for the results.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from sqlalchemy import select

from app.core.config import REPO_ROOT, Settings, get_settings
from app.db.models import Document, Job
from app.db.session import SessionFactory, engine
from app.ingest import queue
from app.ingest.arxiv import ArxivClient, parse_arxiv_id
from app.ingest.pipeline import Ingestor
from app.ingest.storage import SOURCE_TYPES, save_file
from app.llm.embeddings import Embedder

# Loggers that report every table cell they could not place
_NOISY_LOGGERS = ("MatchingPostProcessor", "TFPredictor")


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not verbose:
        for name in _NOISY_LOGGERS:
            # A filter, because these loggers reset their own level when created.
            logging.getLogger(name).addFilter(lambda record: record.levelno >= logging.ERROR)


def print_progress(document: Document, message: str) -> None:
    print(f"  [{document.title}] {message}", flush=True)


def local_path(source: str) -> Path | None:
    """Relative paths are tried against the current directory, then the repo root."""
    for base in (Path.cwd(), REPO_ROOT):
        candidate = base / Path(source).expanduser()
        if candidate.exists():
            return candidate
    return None


def find_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        file
        for file in path.rglob("*")
        if file.is_file()
        and file.suffix.lower() in SOURCE_TYPES
        # Skips hidden folders such as .ipynb_checkpoints
        and not any(part.startswith(".") for part in file.relative_to(path).parts)
    )


async def enqueue(settings: Settings, args: argparse.Namespace) -> tuple[list[int], int]:
    """Returns the queued job ids and the number of sources that could not be added."""
    options = {"ocr": args.ocr, "formulas": not args.no_formulas}
    max_bytes = settings.max_upload_mb * 1024 * 1024
    job_ids: list[int] = []
    problems = 0
    async with SessionFactory() as session:
        for source in args.sources:
            path = local_path(source)
            if path is None:
                try:
                    arxiv_id, version = parse_arxiv_id(source)
                except ValueError:
                    print(f"{source}: not a file, folder or arXiv ID")
                    problems += 1
                    continue
                result = await queue.add_arxiv(session, arxiv_id, version, options, args.force)
                results = [result]
            else:
                files = find_files(path)
                if not files:
                    print(f"{source}: no .pdf or .ipynb files found")
                    problems += 1
                results = []
                for file in files:
                    try:
                        with file.open("rb") as handle:
                            stored = save_file(settings.data_dir, handle, file.name, max_bytes)
                    except ValueError as exc:
                        print(f"{file}: {exc}")
                        problems += 1
                        continue
                    results.append(
                        await queue.add_file(session, stored, file.name, options, args.force)
                    )
            for result in results:
                print(f"{result.document.title}: {result.message}")
                if result.job is not None:
                    job_ids.append(result.job.id)
        await session.commit()
    return job_ids, problems


async def wait_for(job_ids: list[int], poll_seconds: float = 3.0) -> None:
    reported: dict[int, str | None] = {}
    while True:
        async with SessionFactory() as session:
            jobs = list(await session.scalars(select(Job).where(Job.id.in_(job_ids))))
        for job in jobs:
            if reported.get(job.id) != job.progress:
                reported[job.id] = job.progress
                print(f"  [job {job.id}] {job.status}: {job.progress}", flush=True)
        if all(job.status in ("done", "failed") for job in jobs):
            return
        await asyncio.sleep(poll_seconds)


async def summarize(job_ids: list[int]) -> int:
    async with SessionFactory() as session:
        rows = await session.execute(
            select(Job, Document)
            .join(Document, Job.document_id == Document.id)
            .where(Job.id.in_(job_ids))
            .order_by(Job.id)
        )
        failed = 0
        print("\nSummary:")
        for job, document in rows.tuples():
            if job.status == "done":
                details = document.details
                print(
                    f"  ok      {document.title}: {details.get('chunks')} chunks, "
                    f"{details.get('tokens')} tokens, parsed in {details.get('parse_seconds')} s, "
                    f"embedded in {details.get('embed_seconds')} s"
                )
            else:
                failed += 1
                print(f"  {job.status:<7} {document.title}: {job.error or job.progress}")
    return 1 if failed else 0


async def main(args: argparse.Namespace) -> int:
    settings = get_settings()
    if settings.environment != "local":
        print("Ingestion runs locally (ENVIRONMENT=local).")
        return 1
    try:
        job_ids, problems = await enqueue(settings, args)
        if not job_ids:
            return 1 if problems else 0

        async with queue.ingest_lock(engine) as acquired:
            if acquired:
                async with SessionFactory() as session:
                    await queue.requeue_interrupted(session)
                async with (
                    Embedder(settings) as embedder,
                    ArxivClient(settings.data_dir / "arxiv") as arxiv,
                ):
                    ingestor = Ingestor(settings, SessionFactory, embedder, arxiv, print_progress)
                    await ingestor.process_queue(stop_when_empty=True)
            else:
                print("A worker is running and will process the jobs; waiting for it...")
                await wait_for(job_ids)
        return max(await summarize(job_ids), 1 if problems else 0)
    finally:
        await engine.dispose()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "sources",
        nargs="+",
        metavar="SOURCE",
        help="a .pdf or .ipynb file, a folder, or an arXiv ID or URL",
    )
    parser.add_argument("--ocr", action="store_true", help="recognize text in scanned PDF pages")
    parser.add_argument(
        "--no-formulas",
        action="store_true",
        help="skip converting PDF equations to LaTeX (faster)",
    )
    parser.add_argument(
        "--force", action="store_true", help="ingest again even if a document is already ready"
    )
    parser.add_argument("--verbose", action="store_true", help="show library log messages")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    configure_logging(arguments.verbose)
    sys.exit(asyncio.run(main(arguments)))

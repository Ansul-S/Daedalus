"""Process jobs queued through the API until stopped with Ctrl+C.

Run from the repo root:  make worker
or from backend/:        uv run --group ingest python -m scripts.worker

Both kinds of job are handled: ingesting a document and writing a batch of questions.
Ingestion comes first whenever there is any, since somebody is usually waiting on it, and
only one job runs at a time because two sets of models do not fit in 16 GB. A job interrupted
by stopping the worker is queued again the next time it starts.
"""

import argparse
import asyncio
import sys

from app.core.config import Settings, get_settings
from app.db.session import SessionFactory, engine
from app.ingest import queue
from app.ingest.arxiv import ArxivClient
from app.ingest.pipeline import Ingestor
from app.llm.embeddings import Embedder
from app.llm.models import helper_model, paced_generation_model
from app.questions.batch import run_job, spent_today
from scripts.ingest import configure_logging, print_progress

POLL_SECONDS = 2.0


def say(message: str) -> None:
    print(f"  {message}", flush=True)


async def take_a_job(settings: Settings, ingestor: Ingestor, embedder: Embedder) -> bool:
    """Run the next job of either kind. Returns whether there was one."""
    async with SessionFactory() as session:
        job_id = await queue.claim_next_job(session, "ingest")
    if job_id is not None:
        await ingestor.run(job_id)
        return True

    async with SessionFactory() as session:
        job_id = await queue.claim_next_job(session, "generate")
    if job_id is None:
        return False
    print(f"Writing questions for job {job_id}.", flush=True)
    async with SessionFactory() as session:
        spent = await spent_today(session)
    summary = await run_job(
        SessionFactory,
        job_id,
        model=paced_generation_model(settings, spent),
        checker=helper_model(settings),
        embedder=embedder,
        similarity=settings.duplicate_similarity,
        report=say,
    )
    print(f"  job {job_id}: {summary.accepted} accepted, {summary.rejected} rejected", flush=True)
    return True


async def main() -> int:
    settings = get_settings()
    if settings.environment != "local":
        print("Ingestion runs locally (ENVIRONMENT=local).")
        return 1
    try:
        async with queue.ingest_lock(engine) as acquired:
            if not acquired:
                print("Another worker or `make ingest` is already running.")
                return 1
            async with SessionFactory() as session:
                if requeued := await queue.requeue_interrupted(session):
                    print(f"Queued {requeued} interrupted job(s) again.")
            async with (
                Embedder(settings) as embedder,
                ArxivClient(settings.data_dir / "arxiv") as arxiv,
            ):
                ingestor = Ingestor(settings, SessionFactory, embedder, arxiv, print_progress)
                print("Waiting for jobs. Press Ctrl+C to stop.", flush=True)
                while True:
                    if not await take_a_job(settings, ingestor, embedder):
                        await asyncio.sleep(POLL_SECONDS)
    finally:
        await engine.dispose()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--verbose", action="store_true", help="show library log messages and progress bars"
    )
    configure_logging(parser.parse_args().verbose)
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nStopped.")

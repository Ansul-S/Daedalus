"""Process ingestion jobs queued through the API until stopped with Ctrl+C.

Run from the repo root:  make worker
or from backend/:        uv run --group ingest python -m scripts.worker

A job interrupted by stopping the worker is queued again the next time it starts.
"""

import argparse
import asyncio
import sys

from app.core.config import get_settings
from app.db.session import SessionFactory, engine
from app.ingest import queue
from app.ingest.arxiv import ArxivClient
from app.ingest.pipeline import Ingestor
from app.llm.embeddings import Embedder
from scripts.ingest import configure_logging, print_progress


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
                print("Waiting for ingestion jobs. Press Ctrl+C to stop.", flush=True)
                await ingestor.process_queue(stop_when_empty=False)
    finally:
        await engine.dispose()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--verbose", action="store_true", help="show library log messages")
    configure_logging(parser.parse_args().verbose)
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nStopped.")

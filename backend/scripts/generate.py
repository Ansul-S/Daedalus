"""Write a batch of interview questions from the topic map.

Run from the repo root:  make generate N=20
or from backend/:        uv run python -m scripts.generate [OPTIONS]

The plan is stored before any of it runs, so stopping with Ctrl+C loses at most the question
in flight. Start the same job again with --job to carry on where it stopped.
"""

import argparse
import asyncio
import sys

from app.core.checks import check_ollama
from app.core.config import get_settings
from app.db.session import SessionFactory, engine
from app.ingest import queue
from app.llm.embeddings import Embedder
from app.llm.models import helper_model, paced_generation_model
from app.questions.batch import run_job, start_run
from scripts.ingest import configure_logging


def say(message: str) -> None:
    print(f"  {message}", flush=True)


async def main(args: argparse.Namespace) -> int:
    settings = get_settings()
    if settings.environment != "local":
        print("Question generation runs locally (ENVIRONMENT=local).")
        return 1
    # The checker and the duplicate check both run on the local models.
    failed = [check for check in await check_ollama(settings) if check.status == "fail"]
    if failed:
        for check in failed:
            print(f"{check.name}: {check.detail}")
        return 1

    try:
        async with queue.ingest_lock(engine) as acquired:
            if not acquired:
                print("A worker or an ingestion is running; only one at a time fits in memory.")
                return 1
            job_id = args.job
            if job_id is None:
                async with SessionFactory() as session:
                    started = await start_run(session, args.count, document_id=args.document)
                    if started is None:
                        print("Nothing left to ask about; build the topic map with `make topics`.")
                        return 1
                    await session.commit()
                    job, tasks = started
                    job_id, planned = job.id, len(tasks)
                print(f"Job {job_id}: {planned} question(s) planned.")

            async with Embedder(settings) as embedder:
                summary = await run_job(
                    SessionFactory,
                    job_id,
                    model=paced_generation_model(settings),
                    checker=helper_model(settings),
                    embedder=embedder,
                    similarity=settings.duplicate_similarity,
                    report=say,
                )
    finally:
        await engine.dispose()

    print(
        f"\n{summary.accepted} accepted, {summary.rejected} rejected, {summary.failed} failed."
        f"\nCost: {summary.usage}"
    )
    if summary.stopped:
        print(f"Stopped early: {summary.stopped}\nCarry on later with --job {job_id}.")
    return 1 if summary.failed and not summary.written else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--count", type=int, default=10, metavar="N", help="how many questions to write"
    )
    parser.add_argument("--document", type=int, metavar="ID", help="ask about one document only")
    parser.add_argument(
        "--job", type=int, metavar="ID", help="carry on with a job that stopped early"
    )
    parser.add_argument("--verbose", action="store_true", help="show library log messages")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    configure_logging(arguments.verbose)
    sys.exit(asyncio.run(main(arguments)))

"""Build the topic map: tag every chunk with the small local model, then cluster the tags.

Run from the repo root:  make topics
or from backend/:        uv run --group ingest python -m scripts.topics [OPTIONS]

Tagging takes a few seconds per chunk and every chunk is saved as it is tagged, so the
command can be interrupted and started again. Clustering needs the embedding model. Like
ingesting and generating, it runs only when no worker is running: a running worker builds the
map itself when asked through the API (`POST /topics/build`).
"""

import argparse
import asyncio
import sys
from collections import Counter

from app.core.checks import check_ollama
from app.core.config import Settings, get_settings
from app.db.session import SessionFactory, engine
from app.ingest import queue
from app.llm.embeddings import Embedder
from app.llm.models import helper_model
from app.questions.tagging import TaggedChunk, apply_rules, tag_chunks
from app.questions.topics import TopicDraft, build_topic_map, build_topics
from scripts.ingest import configure_logging


def say(message: str) -> None:
    print(f"  {message}", flush=True)


def report_tags(tagged: list[TaggedChunk]) -> None:
    asked = sum(1 for reading in tagged if reading.worth_asking)
    vetoed = Counter(
        reading.skip_reason.split(":")[0]
        for reading in tagged
        if reading.skip_reason is not None and reading.model_worth_asking
    )
    print(f"\nTagged {len(tagged)} chunks; {asked} worth asking about.")
    for reason, count in vetoed.most_common():
        print(f"  {count} the model would have asked about, held back by the rules: {reason}")


def report_topics(topics: list[TopicDraft]) -> None:
    print(f"\nTopics ({len(topics)}):")
    for topic in topics:
        tags = ", ".join(topic.tags[1:])
        print(f"  {topic.chunks:>3} chunks  {topic.name}{f'  [{tags}]' if tags else ''}")


async def build(args: argparse.Namespace, settings: Settings) -> None:
    if not (args.tag_only or args.cluster_only or args.rules_only):
        print(f"Tagging chunks with {settings.helper_model}, then clustering the tags...")
        async with Embedder(settings) as embedder:
            built = await build_topic_map(
                SessionFactory,
                helper_model(settings),
                embedder,
                similarity=args.similarity,
                document_id=args.document,
                limit=args.limit,
                retag=args.retag,
                report=say,
            )
        report_tags(built.tagged)
        report_topics(built.topics)
        return

    # One half of the build, or the rules on their own
    if args.rules_only:
        print("Judging the stored tags by the rules again...")
        await apply_rules(SessionFactory, report=say)
    elif not args.cluster_only:
        print(f"Tagging chunks with {settings.helper_model}...")
        tagged = await tag_chunks(
            SessionFactory,
            helper_model(settings),
            document_id=args.document,
            limit=args.limit,
            retag=args.retag,
            report=say,
        )
        report_tags(tagged)
    if args.tag_only:
        return
    print("\nClustering tags into topics...")
    async with Embedder(settings) as embedder:
        topics = await build_topics(
            SessionFactory, embedder, similarity=args.similarity, report=say
        )
    report_topics(topics)


async def main(args: argparse.Namespace) -> int:
    settings = get_settings()
    if settings.environment != "local":
        print("The topic map is built locally (ENVIRONMENT=local).")
        return 1
    failed = [check for check in await check_ollama(settings) if check.status == "fail"]
    if failed:
        for check in failed:
            print(f"{check.name}: {check.detail}")
        return 1
    try:
        # Tagging and clustering beside a worker would compete for memory, and for the links
        # between chunks and topics that both would be writing.
        async with queue.ingest_lock(engine) as acquired:
            if not acquired:
                print(
                    "A worker or an ingestion is running; only one at a time fits in memory."
                    "\nA running worker builds the map when asked: POST /topics/build."
                )
                return 1
            await build(args, settings)
            return 0
    finally:
        await engine.dispose()


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--document", type=int, metavar="ID", help="tag one document only")
    parser.add_argument("--limit", type=int, metavar="N", help="tag at most N chunks")
    parser.add_argument(
        "--retag", action="store_true", help="tag chunks again that already have tags"
    )
    parser.add_argument("--tag-only", action="store_true", help="skip clustering")
    parser.add_argument(
        "--cluster-only", action="store_true", help="skip tagging and cluster the tags there are"
    )
    parser.add_argument(
        "--rules-only",
        action="store_true",
        help="skip the model and judge the stored tags by the rules again",
    )
    parser.add_argument(
        "--similarity",
        type=float,
        default=settings.topic_similarity,
        metavar="X",
        help=f"how close two tags must be to share a topic (default {settings.topic_similarity})",
    )
    parser.add_argument("--verbose", action="store_true", help="show library log messages")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    configure_logging(arguments.verbose)
    sys.exit(asyncio.run(main(arguments)))

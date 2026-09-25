"""Grouping the chunk tags into topics.

Every distinct tag is embedded once and tags whose vectors sit close together become one
topic, named after the tag that the most chunks used. Clustering runs over the whole library
at once, so the same idea in a paper and in a notebook lands in one topic.

Only chunks worth asking about shape the topics: passages that are scaffolding or plain code
keep their tags but would otherwise add topics like "imports" to the list. A topic keeps its
id across runs as long as its name survives, so questions filed under it stay filed there.

The whole map is built in two halves, tagging the chunks that have no tags yet and then
clustering, by `make topics` and by the worker's topics jobs alike.
"""

import logging
import math
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from pydantic_ai.models import Model
from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import Chunk, ChunkTags, ChunkTopic, Job, Question, Topic
from app.llm.embeddings import Embedder
from app.questions.tagging import TaggedChunk, tag_chunks

log = logging.getLogger(__name__)

Reporter = Callable[[str], None]
# Told what the build is doing before each step, e.g. "tagging passage 3 of 40"
StepReporter = Callable[[str], Awaitable[None]]


@dataclass
class TopicDraft:
    name: str
    tags: list[str]
    embedding: list[float]
    chunks: int


def centroid(vectors: Sequence[Sequence[float]]) -> list[float]:
    means = [sum(values) / len(vectors) for values in zip(*vectors, strict=True)]
    length = math.sqrt(sum(value * value for value in means))
    return [value / length for value in means] if length else means


def group_tags(
    counts: Counter[str], vectors: Sequence[Sequence[float]], similarity: float
) -> list[TopicDraft]:
    """Average-linkage agglomerative clustering, cut where tags stop being the same idea.

    `counts` is how many chunks used each tag, in the order the vectors were embedded.
    """
    tags = list(counts)
    if len(tags) < 2:
        labels = [0] * len(tags)
    else:
        # Imported here: scikit-learn belongs to the ingest group, which the API does without.
        from sklearn.cluster import AgglomerativeClustering

        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - similarity,
            metric="cosine",
            linkage="average",
        ).fit_predict(vectors)

    members: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        members.setdefault(int(label), []).append(index)

    drafts = []
    for group in members.values():
        # The most used tag names the topic; length and spelling break ties, so that the same
        # tags always produce the same name.
        group.sort(key=lambda index: (-counts[tags[index]], len(tags[index]), tags[index]))
        drafts.append(
            TopicDraft(
                name=tags[group[0]],
                tags=[tags[index] for index in group],
                embedding=centroid([vectors[index] for index in group]),
                chunks=sum(counts[tags[index]] for index in group),
            )
        )
    drafts.sort(key=lambda draft: (-draft.chunks, draft.name))
    return drafts


async def tags_worth_asking(session: AsyncSession) -> list[tuple[int, list[str]]]:
    """The tags of every current chunk a question could be asked about."""
    rows = await session.execute(
        select(ChunkTags.chunk_id, ChunkTags.tags)
        .join(Chunk, Chunk.id == ChunkTags.chunk_id)
        .where(Chunk.superseded_at.is_(None), ChunkTags.worth_asking)
        .order_by(ChunkTags.chunk_id)
    )
    return [(chunk_id, tags) for chunk_id, tags in rows.tuples()]


async def save(
    session: AsyncSession, drafts: list[TopicDraft], tagged: list[tuple[int, list[str]]]
) -> dict[str, int]:
    """Write the topics and relink the chunks. Returns the topic id of each name."""
    ids: dict[str, int] = {}
    for draft in drafts:
        statement = (
            insert(Topic)
            .values(name=draft.name, tags=draft.tags, embedding=draft.embedding)
            .on_conflict_do_update(
                index_elements=[Topic.name],
                set_={"tags": draft.tags, "embedding": draft.embedding},
            )
            .returning(Topic.id)
        )
        ids[draft.name] = await session.scalar(statement)

    topic_of_tag = {tag: ids[draft.name] for draft in drafts for tag in draft.tags}
    links = sorted(
        {(chunk_id, topic_of_tag[tag]) for chunk_id, tags in tagged for tag in tags},
    )
    current = select(Chunk.id).where(Chunk.superseded_at.is_(None))
    await session.execute(delete(ChunkTopic).where(ChunkTopic.chunk_id.in_(current)))
    if links:
        await session.execute(
            insert(ChunkTopic),
            [{"chunk_id": chunk_id, "topic_id": topic_id} for chunk_id, topic_id in links],
        )

    # Topics from an earlier run that no chunk and no question refers to any more
    await session.execute(
        delete(Topic).where(
            ~select(ChunkTopic.topic_id).where(ChunkTopic.topic_id == Topic.id).exists(),
            ~select(Question.id).where(Question.topic_id == Topic.id).exists(),
        )
    )
    return ids


async def build_topics(
    sessions: async_sessionmaker[AsyncSession],
    embedder: Embedder,
    *,
    similarity: float,
    report: Reporter | None = None,
) -> list[TopicDraft]:
    say = report or log.info
    async with sessions() as session:
        tagged = await tags_worth_asking(session)
    counts = Counter(tag for _, tags in tagged for tag in tags)
    if not counts:
        say("no tagged chunks worth asking about yet")
        return []

    # Alphabetical, so that the same tags always cluster the same way
    counts = Counter(dict(sorted(counts.items())))
    say(f"embedding {len(counts)} tags from {len(tagged)} chunks")
    vectors = await embedder.embed_documents(list(counts))
    drafts = group_tags(counts, vectors, similarity)

    async with sessions() as session, session.begin():
        await save(session, drafts, tagged)
    say(f"{len(drafts)} topics")
    return drafts


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


@dataclass
class TopicMap:
    # The chunks this build tagged; the ones tagged before are not read again
    tagged: list[TaggedChunk]
    topics: list[TopicDraft]

    @property
    def summary(self) -> str:
        return f"{plural(len(self.tagged), 'passage')} tagged, {plural(len(self.topics), 'topic')}"


async def build_topic_map(
    sessions: async_sessionmaker[AsyncSession],
    model: Model,
    embedder: Embedder,
    *,
    similarity: float,
    document_id: int | None = None,
    limit: int | None = None,
    retag: bool = False,
    step: StepReporter | None = None,
    report: Reporter | None = None,
) -> TopicMap:
    """Tag the chunks that have no tags yet, then cluster every tag in the library into topics.

    Tagging takes the local model a few seconds a chunk, so only new chunks are read unless
    `retag` asks for all of them; clustering always covers the whole library.
    """
    say = report or log.info

    async def tagging(number: int, total: int) -> None:
        if step is not None:
            await step(f"tagging passage {number} of {total}")

    tagged = await tag_chunks(
        sessions,
        model,
        document_id=document_id,
        limit=limit,
        retag=retag,
        progress=tagging,
        report=say,
    )
    say("grouping the tags into topics")
    if step is not None:
        await step("grouping the tags into topics")
    topics = await build_topics(sessions, embedder, similarity=similarity, report=say)
    return TopicMap(tagged=tagged, topics=topics)


async def run_topics_job(
    sessions: async_sessionmaker[AsyncSession],
    job_id: int,
    *,
    model: Model,
    embedder: Embedder,
    similarity: float,
    report: Reporter | None = None,
) -> TopicMap | None:
    """Build the topic map for a claimed job, keeping its progress line current.

    Returns the map, or None when the build failed; the failure is recorded on the job. Every
    chunk tagged before it stays tagged, so building again carries on where this one stopped.
    """
    say = report or log.info

    async def step(message: str) -> None:
        async with sessions() as session, session.begin():
            await session.execute(update(Job).where(Job.id == job_id).values(progress=message))

    try:
        built = await build_topic_map(
            sessions, model, embedder, similarity=similarity, step=step, report=say
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"[:2000]
        log.exception("topics job %s failed", job_id)
        say(f"failed: {message}")
        # The progress line is left at the step that failed.
        async with sessions() as session, session.begin():
            await session.execute(
                update(Job)
                .where(Job.id == job_id)
                .values(status="failed", error=message, finished_at=func.now())
            )
        return None

    async with sessions() as session, session.begin():
        await session.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(status="done", progress=built.summary, error=None, finished_at=func.now())
        )
    return built


async def topic_for(session: AsyncSession, chunk_ids: list[int]) -> int | None:
    """The topic a question written from these chunks belongs under.

    The topic most of the chunks share; among equals, the one covering the most of the
    library, because the broader name is the one a person looks under.
    """
    shared = (
        select(ChunkTopic.topic_id, func.count().label("shared"))
        .where(ChunkTopic.chunk_id.in_(chunk_ids))
        .group_by(ChunkTopic.topic_id)
        .subquery()
    )
    overall = (
        select(ChunkTopic.topic_id, func.count().label("overall"))
        .group_by(ChunkTopic.topic_id)
        .subquery()
    )
    return await session.scalar(
        select(shared.c.topic_id)
        .join(overall, overall.c.topic_id == shared.c.topic_id)
        .join(Topic, Topic.id == shared.c.topic_id)
        .order_by(shared.c.shared.desc(), overall.c.overall.desc(), Topic.name)
        .limit(1)
    )

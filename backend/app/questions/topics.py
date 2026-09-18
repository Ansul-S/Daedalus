"""Grouping the chunk tags into topics.

Every distinct tag is embedded once and tags whose vectors sit close together become one
topic, named after the tag that the most chunks used. Clustering runs over the whole library
at once, so the same idea in a paper and in a notebook lands in one topic.

Only chunks worth asking about shape the topics: passages that are scaffolding or plain code
keep their tags but would otherwise add topics like "imports" to the list. A topic keeps its
id across runs as long as its name survives, so questions filed under it stay filed there.
"""

import logging
import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import Chunk, ChunkTags, ChunkTopic, Question, Topic
from app.llm.embeddings import Embedder

log = logging.getLogger(__name__)

Reporter = Callable[[str], None]


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

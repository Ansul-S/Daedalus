"""The tables that hold the topic map and the generated questions."""

import asyncio

import pytest
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from app.db.models import (
    Chunk,
    ChunkTags,
    ChunkTopic,
    Document,
    Question,
    QuestionSource,
    Topic,
    source_updated,
)

KEY_POINTS = [
    {
        "text": "the dot products grow with the key dimension",
        "weight": 2,
        "evidence_quote": "We divide the dot products by the square root of the key dimension",
        "chunk_id": None,
    }
]


def a_question(**overrides) -> Question:
    fields = {
        "text": "Why are the dot products divided by the square root of the key dimension?",
        "reference_answer": "Their variance grows with the dimension, and large values push "
        "the softmax into a region with tiny gradients.",
        "style": "why_how",
        "difficulty": 3,
        "generator_model": "groq:openai/gpt-oss-120b",
        "prompt_version": "generate-v1",
    }
    return Question(**(fields | overrides))


async def add_question(sessions, chunk_ids: list[int], **overrides) -> int:
    async with sessions() as session, session.begin():
        question = a_question(**overrides)
        session.add(question)
        await session.flush()
        session.add_all(
            QuestionSource(question_id=question.id, chunk_id=chunk_id, position=position)
            for position, chunk_id in enumerate(chunk_ids)
        )
        return question.id


def test_a_question_keeps_its_sources_topic_and_generated_fields(
    sessions, embedder, corpus
) -> None:
    async def scenario():
        async with sessions() as session, session.begin():
            topic = Topic(
                name="attention",
                tags=["attention", "scaled dot-product attention", "softmax scaling"],
                embedding=embedder.vector("attention"),
            )
            session.add(topic)
            await session.flush()
            session.add_all(
                ChunkTopic(chunk_id=chunk_id, topic_id=topic.id)
                for chunk_id in (corpus.scaling, corpus.softmax)
            )
            session.add(
                ChunkTags(
                    chunk_id=corpus.scaling,
                    explains="why attention scores are scaled before the softmax",
                    tags=["attention", "softmax scaling"],
                    worth_asking=True,
                    model_worth_asking=True,
                    model="ollama:qwen3.5:4b",
                    prompt_version="tags-v2",
                )
            )
            topic_id = topic.id

        key_points = [KEY_POINTS[0] | {"chunk_id": corpus.scaling}]
        question_id = await add_question(
            sessions,
            [corpus.scaling, corpus.softmax],
            key_points=key_points,
            misconceptions=["It rescales the vectors to unit length."],
            topic_id=topic_id,
            usage={"requests": 1, "input_tokens": 1506, "output_tokens": 778},
            embedding=embedder.vector("why divide the dot products"),
        )

        async with sessions() as session:
            question = await session.scalar(
                select(Question)
                .where(Question.id == question_id)
                .options(
                    selectinload(Question.topic),
                    selectinload(Question.sources).selectinload(QuestionSource.chunk),
                )
            )
            tags = await session.get_one(ChunkTags, corpus.scaling)
            tagged = await session.scalars(
                select(ChunkTopic.chunk_id).where(ChunkTopic.topic_id == topic_id)
            )
            return question, tags, sorted(tagged)

    question, tags, tagged = asyncio.run(scenario())

    assert question.key_points == [KEY_POINTS[0] | {"chunk_id": corpus.scaling}]
    assert question.misconceptions == ["It rescales the vectors to unit length."]
    assert question.usage["input_tokens"] == 1506
    # Defaults: a question starts out accepted, with an empty validation report
    assert (question.status, question.validation) == ("accepted", {})
    assert question.topic.name == "attention"
    assert [source.position for source in question.sources] == [0, 1]
    assert [source.chunk.id for source in question.sources] == [corpus.scaling, corpus.softmax]
    assert question.sources[0].chunk.section.endswith("Scaled Dot-Product Attention")
    assert (tags.tags, tags.worth_asking, tags.skip_reason) == (
        ["attention", "softmax scaling"],
        True,
        None,
    )
    assert tagged == sorted([corpus.scaling, corpus.softmax])


def test_a_question_is_flagged_when_a_later_ingestion_replaces_its_sources(
    sessions, corpus
) -> None:
    async def scenario():
        replaced = await add_question(sessions, [corpus.scaling, corpus.softmax])
        untouched = await add_question(sessions, [corpus.vanishing])
        async with sessions() as session, session.begin():
            await session.execute(
                update(Chunk).where(Chunk.id == corpus.softmax).values(superseded_at=func.now())
            )
        async with sessions() as session:
            rows = await session.execute(
                select(Question.id, source_updated()).order_by(Question.id)
            )
            return replaced, untouched, rows.all()

    replaced, untouched, flags = asyncio.run(scenario())

    assert flags == [(replaced, True), (untouched, False)]


@pytest.mark.parametrize(
    ("field", "value"),
    [("style", "trivia"), ("status", "draft"), ("difficulty", 0), ("difficulty", 6)],
)
def test_a_question_needs_a_known_style_status_and_difficulty(sessions, corpus, field, value):
    async def scenario():
        async with sessions() as session, session.begin():
            session.add(a_question(**{field: value}))

    with pytest.raises(IntegrityError):
        asyncio.run(scenario())


def test_a_document_cannot_be_deleted_while_a_question_cites_it(sessions, corpus) -> None:
    """Deleting the document cascades to its chunks, and a chunk a question was written from
    may not go: the question could no longer be graded or checked against its sources."""

    async def delete_paper():
        async with sessions() as session, session.begin():
            await session.execute(
                delete(Document).where(Document.arxiv_id == "1706.03762"),
            )

    async def scenario():
        question_id = await add_question(sessions, [corpus.scaling])
        with pytest.raises(IntegrityError):
            await delete_paper()
        async with sessions() as session, session.begin():
            await session.execute(delete(Question).where(Question.id == question_id))
        await delete_paper()
        async with sessions() as session:
            return await session.scalar(select(func.count()).select_from(Chunk))

    assert asyncio.run(scenario()) == 2

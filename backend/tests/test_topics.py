"""Tagging chunks with the local model, and clustering those tags into topics."""

import asyncio
import json

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile
from sqlalchemy import func, select, update

from app.db.models import Chunk, ChunkTags, ChunkTopic, Document, Job, Question, Topic
from app.ingest import queue
from app.questions.tagging import apply_rules, clean_tags, context_only, tag_chunks
from app.questions.topics import build_topics, run_topics_job

READING = {"explains": "why the scores are scaled", "tags": ["Attention"], "worth_asking": True}


def a_chunk(**fields) -> Chunk:
    defaults = {"position": 3, "content_types": ["text"], "text": "...", "token_count": 3}
    return Chunk(**(defaults | fields))


@pytest.mark.parametrize(
    ("chunk", "reason"),
    [
        (a_chunk(section="3.2 Attention"), None),
        (a_chunk(section=None), None),
        (a_chunk(section="Retriever", content_types=["code"]), "code only"),
        (a_chunk(section="Roadmap"), "boilerplate: roadmap"),
        (a_chunk(section="ACKNOWLEDGMENT"), "boilerplate: acknowledgment"),
        (a_chunk(section="Setup and Imports"), "boilerplate: setup"),
        (a_chunk(section="1 Learning Objectives"), "boilerplate: learning objectives"),
        # Deliberately narrow: a training objective is a subject, not scaffolding.
        (a_chunk(section="Optimization Objectives"), None),
        # Every section a chunk covers has to be scaffolding before the chunk is.
        (a_chunk(section="Setup · Imports"), "boilerplate: setup"),
        (a_chunk(section="V. CONCLUSION · VI. ACKNOWLEDGMENT"), None),
        (a_chunk(position=0, section=None), "front matter"),
    ],
)
def test_code_and_scaffolding_can_only_be_context(chunk: Chunk, reason: str | None) -> None:
    assert context_only(chunk) == reason


def test_tags_are_folded_together_and_capped() -> None:
    tags = clean_tags(
        ["Dot-Product Attention", "dot product attention", " softmax. ", "", "a", "b", "c", "d"]
    )

    assert tags == ["dot product attention", "softmax", "a", "b", "c"]


def tagger_model(calls: list[str]) -> FunctionModel:
    """A stand-in for the local model: it records each passage and always says the same thing."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(str(messages[-1].parts[-1].content))
        return ModelResponse(parts=[TextPart(json.dumps(READING))])

    return FunctionModel(respond, profile=ModelProfile(supports_json_schema_output=True))


@pytest.fixture
def lesson(sessions, embedder) -> list[int]:
    """A notebook of three chunks: prose, code only, and a setup section."""
    rows = [
        ("Retriever", ["text"], "FAISS ranks the chunks by cosine distance."),
        ("Retriever", ["code"], "index = faiss.IndexFlatIP(384)"),
        ("Setup", ["text"], "Install the packages listed in requirements.txt."),
    ]

    async def add() -> list[int]:
        async with sessions() as session, session.begin():
            notebook = Document(source_type="notebook", title="Retrieval QnA", status="ready")
            notebook.chunks = [
                Chunk(
                    position=position,
                    section=section,
                    content_types=content_types,
                    text=body,
                    token_count=len(body.split()),
                    embedding=embedder.vector(body),
                )
                for position, (section, content_types, body) in enumerate(rows)
            ]
            session.add(notebook)
            await session.flush()
            return [chunk.id for chunk in notebook.chunks]

    return asyncio.run(add())


def test_tagging_records_what_the_model_and_the_rules_decided(sessions, lesson) -> None:
    calls: list[str] = []

    async def scenario():
        tagged = await tag_chunks(sessions, tagger_model(calls), report=lambda message: None)
        async with sessions() as session:
            stored = await session.scalars(select(ChunkTags).order_by(ChunkTags.chunk_id))
            return tagged, list(stored)

    tagged, stored = asyncio.run(scenario())

    assert [reading.chunk_id for reading in tagged] == lesson
    assert [(row.worth_asking, row.model_worth_asking, row.skip_reason) for row in stored] == [
        (True, True, None),
        (False, True, "code only"),
        (False, True, "boilerplate: setup"),
    ]
    assert [row.tags for row in stored] == [["attention"]] * 3
    assert stored[0].explains == "why the scores are scaled"
    assert (stored[0].prompt_version, stored[0].model) == ("tags-v2", "function:respond:")
    # The model is shown the document, the section and the passage, in delimiters.
    assert "Document: Retrieval QnA" in calls[0]
    assert "Section: Retriever" in calls[0]
    assert "<<<\nFAISS ranks the chunks by cosine distance.\n>>>" in calls[0]


def test_tagging_only_visits_current_chunks_that_have_none(sessions, lesson) -> None:
    async def scenario():
        calls: list[str] = []
        await tag_chunks(sessions, tagger_model(calls), limit=1, report=lambda _: None)
        async with sessions() as session, session.begin():
            await session.execute(
                update(Chunk).where(Chunk.id == lesson[1]).values(superseded_at=func.now())
            )
        second: list[str] = []
        await tag_chunks(sessions, tagger_model(second), report=lambda _: None)
        third: list[str] = []
        await tag_chunks(sessions, tagger_model(third), report=lambda _: None)
        retagged: list[str] = []
        await tag_chunks(sessions, tagger_model(retagged), retag=True, report=lambda _: None)
        async with sessions() as session:
            tagged = await session.scalars(select(ChunkTags.chunk_id).order_by(ChunkTags.chunk_id))
            return len(calls), len(second), len(third), len(retagged), list(tagged)

    first, second, third, retagged, tagged = asyncio.run(scenario())

    assert first == 1
    # The superseded chunk is left out, and the one already tagged is not tagged again.
    assert second == 1
    assert third == 0
    assert retagged == 2
    assert tagged == [lesson[0], lesson[2]]


def test_tagging_says_which_chunk_it_is_on(sessions, lesson) -> None:
    calls: list[str] = []
    heard: list[tuple[int, int, int]] = []

    async def progress(number: int, total: int) -> None:
        heard.append((number, total, len(calls)))

    asyncio.run(tag_chunks(sessions, tagger_model(calls), progress=progress, report=lambda _: None))

    # Each chunk is announced before the model reads it
    assert heard == [(1, 3, 0), (2, 3, 1), (3, 3, 2)]


def test_the_rules_can_be_judged_again_without_the_model(sessions, lesson) -> None:
    """Changing the rules must not cost another pass over the library with the local model."""

    async def scenario():
        await tag_chunks(sessions, tagger_model([]), report=lambda _: None)
        # As if the rules had stopped counting this section as scaffolding
        async with sessions() as session, session.begin():
            await session.execute(
                update(Chunk).where(Chunk.id == lesson[2]).values(section="Ranking")
            )
        changed = await apply_rules(sessions, report=lambda _: None)
        async with sessions() as session:
            stored = await session.scalars(select(ChunkTags).order_by(ChunkTags.chunk_id))
            return changed, list(stored)

    changed, stored = asyncio.run(scenario())

    assert [reading.chunk_id for reading in changed] == [lesson[2]]
    assert [(row.worth_asking, row.skip_reason) for row in stored] == [
        (True, None),
        (False, "code only"),
        (True, None),
    ]
    # What the model said is kept; only the rules were judged again.
    assert [row.model_worth_asking for row in stored] == [True, True, True]


async def tag(sessions, chunk_id: int, tags: list[str], worth_asking: bool = True) -> None:
    async with sessions() as session, session.begin():
        session.add(
            ChunkTags(
                chunk_id=chunk_id,
                explains="something" if worth_asking else "nothing",
                tags=tags,
                worth_asking=worth_asking,
                model_worth_asking=worth_asking,
                model="test",
                prompt_version="tags-v2",
            )
        )


def test_topics_gather_close_tags_from_every_document(sessions, embedder, corpus) -> None:
    async def scenario():
        await tag(sessions, corpus.scaling, ["attention", "softmax"])
        await tag(sessions, corpus.retriever, ["attention"])
        await tag(sessions, corpus.vanishing, ["self attention"])
        await tag(sessions, corpus.positions, ["roadmap"], worth_asking=False)
        drafts = await build_topics(sessions, embedder, similarity=0.7, report=lambda _: None)
        async with sessions() as session:
            rows = await session.execute(
                select(ChunkTopic.chunk_id, Topic.name)
                .join(Topic, Topic.id == ChunkTopic.topic_id)
                .order_by(ChunkTopic.chunk_id, Topic.name)
            )
            return drafts, rows.all()

    drafts, links = asyncio.run(scenario())

    # "self attention" is near enough to "attention" to share a topic, which the tag used by
    # the most chunks names; "softmax" is on its own and "roadmap" never becomes a topic.
    assert [(draft.name, draft.tags, draft.chunks) for draft in drafts] == [
        ("attention", ["attention", "self attention"], 3),
        ("softmax", ["softmax"], 1),
    ]
    assert links == [
        (corpus.scaling, "attention"),
        (corpus.scaling, "softmax"),
        (corpus.vanishing, "attention"),
        (corpus.retriever, "attention"),
    ]


def test_rebuilding_topics_keeps_the_one_a_question_points_at(sessions, embedder, corpus) -> None:
    async def scenario():
        await tag(sessions, corpus.scaling, ["attention"])
        await tag(sessions, corpus.vanishing, ["softmax"])
        await build_topics(sessions, embedder, similarity=0.9, report=lambda _: None)
        async with sessions() as session, session.begin():
            before = dict(
                (await session.execute(select(Topic.name, Topic.id))).tuples().all()  # noqa: C408
            )
            session.add(
                Question(
                    text="Why is the softmax flattened?",
                    reference_answer="Because the logits are divided by a temperature.",
                    style="why_how",
                    difficulty=2,
                    topic_id=before["softmax"],
                    generator_model="test",
                    prompt_version="generate-v1",
                )
            )
        # The chunk that carried "softmax" now says something else.
        async with sessions() as session, session.begin():
            await session.execute(
                update(ChunkTags).where(ChunkTags.chunk_id == corpus.vanishing).values(tags=["rrf"])
            )
        await build_topics(sessions, embedder, similarity=0.9, report=lambda _: None)
        async with sessions() as session:
            after = dict((await session.execute(select(Topic.name, Topic.id))).tuples().all())  # noqa: C408
            question = await session.scalar(select(Question))
            orphaned = await session.scalar(
                select(func.count())
                .select_from(ChunkTopic)
                .where(ChunkTopic.topic_id == before["softmax"])
            )
            return before, after, question.topic_id, orphaned

    before, after, topic_id, orphaned = asyncio.run(scenario())

    assert sorted(before) == ["attention", "softmax"]
    # "softmax" lost its chunk but a question still files under it, so the row stays put.
    assert sorted(after) == ["attention", "rrf", "softmax"]
    assert after["attention"] == before["attention"]
    assert topic_id == before["softmax"]
    assert orphaned == 0


async def claimed_topics_job(sessions) -> int:
    """A topics job as the worker holds it: queued through the API, then claimed."""
    async with sessions() as session, session.begin():
        session.add(Job(kind="topics"))
    async with sessions() as session:
        return await queue.claim_next_job(session, "topics")


def test_a_topics_job_tags_the_new_chunks_then_groups_the_tags(sessions, embedder, corpus) -> None:
    lines: list[str | None] = []

    async def scenario():
        await tag(sessions, corpus.retriever, ["faiss index"])
        job_id = await claimed_topics_job(sessions)

        async def job_line() -> None:
            async with sessions() as session:
                lines.append(await session.scalar(select(Job.progress).where(Job.id == job_id)))

        # The model and the embedder both look at the job's line when they are called.
        async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            await job_line()
            return ModelResponse(parts=[TextPart(json.dumps(READING))])

        embed = embedder.embed_documents

        async def embed_documents(texts, progress=None):
            await job_line()
            return await embed(texts, progress)

        embedder.embed_documents = embed_documents
        model = FunctionModel(respond, profile=ModelProfile(supports_json_schema_output=True))
        built = await run_topics_job(
            sessions, job_id, model=model, embedder=embedder, similarity=0.8, report=lambda _: None
        )
        async with sessions() as session:
            job = await session.get_one(Job, job_id)
            links = await session.scalar(select(func.count()).select_from(ChunkTopic))
        return built, job, links

    built, job, links = asyncio.run(scenario())

    # The chunk tagged before is not read again, and each line is up before its step starts.
    assert lines == [
        "tagging passage 1 of 4",
        "tagging passage 2 of 4",
        "tagging passage 3 of 4",
        "tagging passage 4 of 4",
        "grouping the tags into topics",
    ]
    assert [(topic.name, topic.chunks) for topic in built.topics] == [
        ("attention", 4),
        ("faiss index", 1),
    ]
    assert (job.status, job.progress, job.error) == ("done", "4 passages tagged, 2 topics", None)
    assert job.finished_at is not None
    assert links == 5


def test_a_failed_topics_job_keeps_its_tags_and_carries_on_next_time(
    sessions, embedder, corpus
) -> None:
    async def scenario():
        embedder.fail = True
        first = await claimed_topics_job(sessions)
        read_first: list[str] = []
        failed = await run_topics_job(
            sessions,
            first,
            model=tagger_model(read_first),
            embedder=embedder,
            similarity=0.8,
            report=lambda _: None,
        )
        async with sessions() as session:
            kept = await session.scalar(select(func.count()).select_from(ChunkTags))

        embedder.fail = False
        second = await claimed_topics_job(sessions)
        read_second: list[str] = []
        built = await run_topics_job(
            sessions,
            second,
            model=tagger_model(read_second),
            embedder=embedder,
            similarity=0.8,
            report=lambda _: None,
        )
        async with sessions() as session:
            jobs = list(await session.scalars(select(Job).order_by(Job.id)))
        return failed, len(read_first), kept, built, len(read_second), jobs

    failed, read_first, kept, built, read_second, jobs = asyncio.run(scenario())

    # The embedding model was away when the tags were to be grouped.
    assert failed is None
    assert (jobs[0].status, jobs[0].error) == (
        "failed",
        "EmbeddingError: Ollama is not reachable (ConnectError)",
    )
    # The line says where it stopped.
    assert jobs[0].progress == "grouping the tags into topics"
    assert jobs[0].finished_at is not None
    # The tags it paid for were kept, so the next build only groups them.
    assert (read_first, kept, read_second) == (5, 5, 0)
    assert [(topic.name, topic.chunks) for topic in built.topics] == [("attention", 5)]
    assert (jobs[1].status, jobs[1].progress) == ("done", "0 passages tagged, 1 topic")

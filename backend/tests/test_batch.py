"""Planning a batch of questions, working through it, and picking it up again."""

import asyncio
import json
import re
from collections import Counter
from datetime import UTC, datetime, timedelta

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile
from sqlalchemy import func, select, update

from app.db.models import Chunk, ChunkTags, ChunkTopic, Job, Question, QuestionTask, Topic
from app.llm.pacing import QuotaExhausted
from app.questions.batch import (
    TaskPlan,
    candidates,
    failure_passages,
    is_about_failure,
    partner_for,
    plan_tasks,
    run_job,
    spent_today,
    start_run,
)

# (topic, chunk, document). Topic 1 spans all three sources, topic 4 sits in one.
FOUND = [
    (1, 10, 1),
    (1, 11, 1),
    (1, 12, 2),
    (1, 13, 3),
    (2, 20, 1),
    (2, 21, 2),
    (3, 30, 1),
    (3, 31, 1),
    (4, 40, 3),
]

# The shape of the real library: the biggest topics all belong to one source
LOPSIDED = (
    [(1, chunk, 1) for chunk in (10, 11, 12, 13, 14)]
    + [(2, chunk, 1) for chunk in (20, 21, 22, 23)]
    + [(3, chunk, 1) for chunk in (30, 31, 32)]
    + [(4, chunk, 2) for chunk in (40, 41, 42)]
    + [(5, chunk, 3) for chunk in (50, 51, 52)]
)

# Six topics, each holding one passage from either source
CROSSING = [(topic, topic * 10 + offset, offset + 1) for topic in range(1, 7) for offset in (0, 1)]

# One source, seven topics of one passage each
SINGLES = [(topic, topic * 10, 1) for topic in range(1, 8)]


def test_passages_are_taken_a_source_and_a_topic_at_a_time() -> None:
    # Every passage could take any style, so the rotation shows in full.
    plans = plan_tasks(FOUND, count=5, failures={chunk for _, chunk, _ in FOUND})
    document_of = {chunk: document for _, chunk, document in FOUND}

    # Every source is asked about in turn, and within one the topic ring decides
    assert [document_of[plan.chunk_ids[0]] for plan in plans] == [1, 2, 3, 1, 2]
    assert [plan.chunk_ids[0] for plan in plans] == [10, 21, 13, 20, 12]
    assert [plan.style for plan in plans] == [
        "why_how",
        "intuition",
        "compare",
        "tradeoffs",
        "failure_modes",
    ]


def test_the_plan_is_spread_over_the_sources() -> None:
    plans = plan_tasks(LOPSIDED, count=9)
    document_of = {chunk: document for _, chunk, document in LOPSIDED}

    # Biggest topic first alone would have given the first source six of the nine, since its
    # three topics are the biggest in the library and the other two sources hold one each.
    assert Counter(document_of[plan.chunk_ids[0]] for plan in plans) == {1: 4, 2: 3, 3: 2}


def test_a_comparison_takes_a_second_passage_from_the_same_topic() -> None:
    plans = plan_tasks(FOUND, count=5)

    # The comparison skips topic 4, which holds one passage, for a topic that can spare two.
    [paired] = [plan for plan in plans if len(plan.chunk_ids) == 2]
    assert (paired.style, paired.chunk_ids) == ("compare", [13, 11])
    # A style that wants a partner and finds none asks a plain question instead.
    assert all(
        plan.style not in ("compare", "connection") for plan in plans if len(plan.chunk_ids) == 1
    )


def test_a_connection_question_joins_two_sources() -> None:
    # "connection" falls sixth in the rotation
    plans = plan_tasks(CROSSING, count=6)
    document_of = {chunk: document for _, chunk, document in CROSSING}

    joined = plans[5]
    assert joined.style == "connection"
    assert [document_of[chunk] for chunk in joined.chunk_ids] == [2, 1]


@pytest.mark.parametrize(
    ("style", "rest", "expected"),
    [
        ("connection", [(11, 1), (12, 2)], 12),
        ("connection", [(11, 1)], None),
        ("compare", [(11, 1)], 11),
        ("why_how", [(11, 1)], None),
    ],
)
def test_a_second_passage_is_chosen_for_the_styles_that_want_one(style, rest, expected) -> None:
    assert partner_for(style, rest, document_id=1) == expected


@pytest.mark.parametrize(
    ("failures", "fifth"),
    [
        # It passes over the next two passages for the one about something going wrong...
        ({70}, TaskPlan(chunk_ids=[70], style="failure_modes")),
        # ...and with none to hand asks a plain question of the next passage instead.
        (set(), TaskPlan(chunk_ids=[50], style="why_how")),
    ],
)
def test_failure_modes_is_asked_only_of_a_passage_about_a_failure(failures, fifth) -> None:
    """Asked of any passage, the style asked what goes wrong when something is left out, and
    the passages mostly never say."""
    plans = plan_tasks(SINGLES, count=5, failures=failures)

    assert [plan.style for plan in plans[:4]] == ["why_how", "intuition", "why_how", "tradeoffs"]
    assert plans[4] == fifth


@pytest.mark.parametrize(
    ("explains", "tags", "expected"),
    [
        ("describes limitations of simple RNNs regarding long-term memory", ["rnn"], True),
        ("identifies gaps in current AI and RAG approaches for software testing", [], True),
        # The tag says what the summary does not.
        ("defines scaled dot-product attention", ["softmax", "vanishing gradient"], True),
        ("defines vocabulary, tokens, and tokenization", ["tokenization"], False),
        ("describes a five-layer prompt architecture", ["prompt engineering"], False),
    ],
)
def test_a_passage_is_about_a_failure_when_its_tagging_says_so(explains, tags, expected) -> None:
    assert is_about_failure(explains, tags) == expected


def test_planning_stops_when_the_library_runs_out() -> None:
    plans = plan_tasks([(1, 10, 1)], count=5)

    assert plans == [TaskPlan(chunk_ids=[10], style="why_how")]


@pytest.fixture
def library(sessions, embedder, corpus):
    """The corpus tagged and filed under topics, ready to be asked about."""

    async def prepare() -> dict[str, int]:
        async with sessions() as session, session.begin():
            topics = [
                Topic(name=name, tags=[name], embedding=embedder.vector(name))
                for name in ("attention", "retrieval")
            ]
            session.add_all(topics)
            await session.flush()
            attention, retrieval = (topic.id for topic in topics)
            for chunk_id in (corpus.scaling, corpus.positions, corpus.softmax, corpus.vanishing):
                session.add(
                    ChunkTags(
                        chunk_id=chunk_id,
                        explains="something",
                        tags=["attention"],
                        worth_asking=True,
                        model_worth_asking=True,
                        model="test",
                        prompt_version="tags-v2",
                    )
                )
                session.add(ChunkTopic(chunk_id=chunk_id, topic_id=attention))
            # Tagged, but not worth asking about
            session.add(
                ChunkTags(
                    chunk_id=corpus.retriever,
                    explains="nothing",
                    tags=["retrieval"],
                    worth_asking=False,
                    model_worth_asking=True,
                    model="test",
                    prompt_version="tags-v2",
                )
            )
            session.add(ChunkTopic(chunk_id=corpus.retriever, topic_id=retrieval))
            return {"attention": attention, "retrieval": retrieval}

    return asyncio.run(prepare())


def test_only_chunks_worth_asking_about_are_candidates(sessions, corpus, library) -> None:
    async def scenario():
        async with sessions() as session, session.begin():
            await session.execute(
                update(Chunk).where(Chunk.id == corpus.softmax).values(superseded_at=func.now())
            )
        async with sessions() as session:
            return await candidates(session)

    found = asyncio.run(scenario())

    # The superseded one and the one not worth asking about are both left out.
    assert sorted(chunk for _, chunk, _ in found) == sorted(
        [corpus.scaling, corpus.positions, corpus.vanishing]
    )


def test_the_passages_about_a_failure_are_read_from_their_tagging(sessions, corpus, library):
    async def scenario():
        async with sessions() as session, session.begin():
            await session.execute(
                update(ChunkTags)
                .where(ChunkTags.chunk_id == corpus.vanishing)
                .values(explains="why a simple RNN fails to carry information far back")
            )
        async with sessions() as session:
            return await failure_passages(
                session, [corpus.scaling, corpus.positions, corpus.vanishing]
            )

    assert asyncio.run(scenario()) == {corpus.vanishing}


def test_a_library_with_nothing_left_to_ask_writes_nothing_down(sessions, corpus) -> None:
    """An empty job would sit queued, and hold up the next batch asked for through the API."""

    async def scenario():
        # The corpus is there but was never tagged, so no passage is a candidate.
        async with sessions() as session:
            started = await start_run(session, 3)
            await session.commit()
        async with sessions() as session:
            return started, await session.scalar(select(func.count()).select_from(Job))

    assert asyncio.run(scenario()) == (None, 0)


def test_a_batch_knows_what_was_spent_in_the_last_day(sessions) -> None:
    """Read at the start of every batch, so that the day's budget is the day's."""

    def written(model: str, requests: int, tokens: int, days_ago: float = 0.0) -> Question:
        return Question(
            text="Why?",
            reference_answer="Because.",
            style="why_how",
            difficulty=3,
            status="accepted",
            generator_model=model,
            prompt_version="generate-v3",
            usage={"requests": requests, "input_tokens": tokens - 100, "output_tokens": 100},
            created_at=datetime.now(UTC) - timedelta(days=days_ago),
        )

    async def scenario():
        async with sessions() as session, session.begin():
            session.add_all(
                [
                    written("openai/gpt-oss-120b", 2, 4_000),
                    written("openai/gpt-oss-120b", 1, 2_500),
                    written("gemini-3.5-flash", 1, 3_000),
                    # Yesterday's batch has had its day.
                    written("openai/gpt-oss-120b", 5, 20_000, days_ago=1.5),
                ]
            )
        async with sessions() as session:
            return await spent_today(session)

    assert asyncio.run(scenario()) == {
        "openai/gpt-oss-120b": (3, 6_500),
        "gemini-3.5-flash": (1, 3_000),
    }


def a_question(messages: list[ModelMessage]) -> ModelResponse:
    """Quote the passage that was given, so the question is grounded by construction and
    worded differently from every other one in the run."""
    prompt = str(messages[-1].parts[-1].content)
    chunk_id = int(re.search(r"\[chunk (\d+)\]", prompt).group(1))
    passage = re.search(r"<<<\n(.*?)\n>>>", prompt, re.S).group(1)
    quote = " ".join(passage.split()[:10])
    point = {"text": "a point", "weight": 2, "evidence_quote": quote, "chunk_id": chunk_id}
    return ModelResponse(
        parts=[
            TextPart(
                json.dumps(
                    {
                        "question": f"Why is it that {quote}?",
                        "difficulty": 3,
                        "reference_answer": f"Because {quote}.",
                        "key_points": [point, point | {"weight": 1}],
                        "misconceptions": [],
                        "source_chunk_ids": [chunk_id],
                    }
                )
            )
        ]
    )


def writer_model(raises: Exception | None = None) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if raises is not None:
            raise raises
        return a_question(messages)

    return FunctionModel(respond, profile=ModelProfile(supports_json_schema_output=True))


def checker_model(**overrides) -> FunctionModel:
    verdict = {
        "kind": "explain",
        "answer": "Because of what the passage says.",
        "answerable": True,
        "missing": "nothing",
    } | overrides

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(json.dumps(verdict))])

    return FunctionModel(respond, profile=ModelProfile(supports_json_schema_output=True))


async def go(sessions, embedder, job_id, writer=None, checker=None):
    return await run_job(
        sessions,
        job_id,
        model=writer or writer_model(),
        checker=checker or checker_model(),
        embedder=embedder,
        similarity=0.7,
        report=lambda message: None,
    )


def test_a_run_writes_every_planned_question_and_files_it(sessions, embedder, library) -> None:
    async def scenario():
        async with sessions() as session:
            job, tasks = await start_run(session, 3)
            await session.commit()
            job_id = job.id
        summary = await go(sessions, embedder, job_id)
        async with sessions() as session:
            job = await session.get_one(Job, job_id)
            stored = list(
                await session.scalars(select(QuestionTask).order_by(QuestionTask.position))
            )
            questions = list(await session.scalars(select(Question)))
            return summary, job, stored, questions

    summary, job, tasks, questions = asyncio.run(scenario())

    assert (summary.written, summary.accepted, summary.failed) == (3, 3, 0)
    assert [task.status for task in tasks] == ["done"] * 3
    assert all(task.question_id is not None for task in tasks)
    assert all(task.attempts == 1 for task in tasks)
    assert (job.status, job.progress) == ("done", "3 accepted, 0 rejected")
    assert job.error is None
    assert len(questions) == 3
    # Every question is filed under the topic its chunk belongs to, and the style it was asked in.
    assert {question.topic_id for question in questions} == {library["attention"]}
    assert {question.id: question.style for question in questions} == {
        task.question_id: task.style for task in tasks
    }
    assert sorted(task.style for task in tasks) == ["compare", "intuition", "why_how"]


def test_a_run_says_which_question_it_is_writing(sessions, embedder, library) -> None:
    """A batch takes minutes, so the job has to say where it is while it is still going."""

    async def scenario():
        async with sessions() as session:
            job, _ = await start_run(session, 3)
            await session.commit()
            job_id = job.id
        writing, carry_on = asyncio.Event(), asyncio.Event()

        async def slowly(messages, info):
            writing.set()
            await carry_on.wait()
            return a_question(messages)

        model = FunctionModel(slowly, profile=ModelProfile(supports_json_schema_output=True))
        run = asyncio.ensure_future(go(sessions, embedder, job_id, writer=model))
        await writing.wait()
        async with sessions() as session:
            midway = (await session.get_one(Job, job_id)).progress
        carry_on.set()
        await run
        async with sessions() as session:
            return midway, (await session.get_one(Job, job_id)).progress

    midway, finished = asyncio.run(scenario())

    assert midway == "writing question 1 of 3"
    assert finished == "3 accepted, 0 rejected"


def test_the_days_budget_running_out_leaves_the_rest_for_later(sessions, embedder, library):
    async def scenario():
        async with sessions() as session:
            job, _ = await start_run(session, 3)
            await session.commit()
            job_id = job.id
        summary = await go(
            sessions, embedder, job_id, writer=writer_model(QuotaExhausted("groq", "out of tokens"))
        )
        async with sessions() as session:
            job = await session.get_one(Job, job_id)
            statuses = list(
                await session.scalars(select(QuestionTask.status).order_by(QuestionTask.position))
            )
            return summary, job, statuses

    summary, job, statuses = asyncio.run(scenario())

    # The first task stops the run, and nothing is marked failed: the work is only unfinished.
    assert (summary.written, summary.failed) == (0, 0)
    assert summary.stopped.startswith("QuotaExhausted")
    assert statuses == ["queued"] * 3
    assert job.status == "queued"
    assert job.error.startswith("QuotaExhausted")


def test_a_run_picks_up_where_it_stopped(sessions, embedder, library) -> None:
    async def scenario():
        async with sessions() as session:
            job, tasks = await start_run(session, 3)
            await session.commit()
            job_id, first = job.id, tasks[0].id
        async with sessions() as session, session.begin():
            await session.execute(
                update(QuestionTask).where(QuestionTask.id == first).values(status="done")
            )
        summary = await go(sessions, embedder, job_id)
        async with sessions() as session:
            written = await session.scalar(select(func.count()).select_from(Question))
            return summary, written

    summary, written = asyncio.run(scenario())

    # The finished task is left alone, so only two questions are paid for.
    assert summary.written == 2
    assert written == 2


def test_a_task_that_fails_is_recorded_and_the_rest_carry_on(sessions, embedder, library) -> None:
    calls: list[int] = []

    def flaky(messages, info):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("the provider fell over")
        return a_question(messages)

    async def scenario():
        async with sessions() as session:
            job, _ = await start_run(session, 3)
            await session.commit()
            job_id = job.id
        model = FunctionModel(flaky, profile=ModelProfile(supports_json_schema_output=True))
        summary = await go(sessions, embedder, job_id, writer=model)
        async with sessions() as session:
            job = await session.get_one(Job, job_id)
            tasks = list(
                await session.scalars(select(QuestionTask).order_by(QuestionTask.position))
            )
            return summary, job, tasks

    summary, job, tasks = asyncio.run(scenario())

    assert (summary.written, summary.failed) == (2, 1)
    assert [task.status for task in tasks] == ["failed", "done", "done"]
    assert tasks[0].error.startswith("RuntimeError: the provider fell over")
    assert job.status == "done"

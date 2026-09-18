"""Planning a batch of questions and working through it.

The plan is written down as tasks before any of it runs, so a batch that stops -- on Ctrl+C,
on a provider running out for the day, or on a failure -- carries on from where it stopped
instead of spending the tokens again. Each task is committed as it finishes.

Chunks are picked a topic at a time, so twenty questions do not all come from one corner of
the library, and the style rotates so they are not all "why does this work". A chunk that an
accepted question already covers is left out, which makes a second run break new ground.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.models import Model
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import Chunk, ChunkTags, ChunkTopic, Document, Job, Question, QuestionSource
from app.db.models import QuestionTask as Task
from app.llm.pacing import QuotaExhausted
from app.questions.generation import Source, generate_question, source_of
from app.questions.topics import topic_for
from app.questions.validation import store_question, validate

log = logging.getLogger(__name__)

# Styles are handed out in this order, one per question
ROTATION = (
    "why_how",
    "intuition",
    "compare",
    "tradeoffs",
    "failure_modes",
    "connection",
    "paper",
)
# Styles that need a second passage to be worth asking
PAIRED = {"compare", "connection"}
FALLBACK_STYLE = "why_how"

Reporter = Callable[[str], None]


@dataclass
class TaskPlan:
    chunk_ids: list[int]
    style: str


@dataclass
class Summary:
    written: int = 0
    accepted: int = 0
    rejected: int = 0
    failed: int = 0
    # Why the run stopped early, if it did
    stopped: str | None = None
    usage: dict[str, int] = field(default_factory=dict)

    def add_usage(self, usage: dict[str, int]) -> None:
        for name, value in usage.items():
            self.usage[name] = self.usage.get(name, 0) + value


async def candidates(
    session: AsyncSession, *, document_id: int | None = None
) -> list[tuple[int, int, int]]:
    """(topic, chunk, document) for every current chunk worth asking about that no accepted
    question covers yet."""
    covered = (
        select(QuestionSource.chunk_id)
        .join(Question, Question.id == QuestionSource.question_id)
        .where(Question.status == "accepted")
    )
    query = (
        select(ChunkTopic.topic_id, Chunk.id, Chunk.document_id)
        .join(Chunk, Chunk.id == ChunkTopic.chunk_id)
        .join(ChunkTags, ChunkTags.chunk_id == Chunk.id)
        .where(Chunk.superseded_at.is_(None), ChunkTags.worth_asking, Chunk.id.not_in(covered))
        .order_by(ChunkTopic.topic_id, Chunk.id)
    )
    if document_id is not None:
        query = query.where(Chunk.document_id == document_id)
    rows = await session.execute(query)
    return [(topic, chunk, document) for topic, chunk, document in rows.tuples()]


def partner_for(style: str, rest: list[tuple[int, int]], document_id: int) -> int | None:
    """A second passage for the styles that want one: another document for a question that
    joins two ideas up, any other passage on the topic for a comparison."""
    if style == "connection":
        return next((chunk for chunk, document in rest if document != document_id), None)
    if style == "compare":
        return rest[0][0] if rest else None
    return None


def plan_tasks(found: list[tuple[int, int, int]], count: int) -> list[TaskPlan]:
    """Take a chunk from each topic in turn, biggest topic first, until the count is met.

    A style that wants two passages picks the next topic that can actually spare a second
    one, rather than taking whichever topic came up and quietly dropping to a plain question:
    most topics cover a single chunk, so the blind rotation almost never paired anything.
    """
    by_topic: dict[int, list[tuple[int, int]]] = {}
    for topic, chunk, document in found:
        by_topic.setdefault(topic, []).append((chunk, document))
    order = sorted(by_topic, key=lambda topic: (-len(by_topic[topic]), topic))

    taken: set[int] = set()
    plans: list[TaskPlan] = []
    cursor = 0

    def free_of(topic: int) -> list[tuple[int, int]]:
        return [pair for pair in by_topic[topic] if pair[0] not in taken]

    def next_topic(style: str | None) -> tuple[int, int] | None:
        """The next topic round the ring with something to offer, and where to look next."""
        for step in range(len(order)):
            topic = order[(cursor + step) % len(order)]
            free = free_of(topic)
            if not free:
                continue
            if style is not None and partner_for(style, free[1:], free[0][1]) is None:
                continue
            return topic, (cursor + step + 1) % len(order)
        return None

    while len(plans) < count and order:
        style = ROTATION[len(plans) % len(ROTATION)]
        found_topic = next_topic(style) if style in PAIRED else None
        if found_topic is None:
            if style in PAIRED:
                style = FALLBACK_STYLE
            found_topic = next_topic(None)
        if found_topic is None:
            # Every topic is used up, so the library cannot fill the count asked for.
            break
        topic, cursor = found_topic
        free = free_of(topic)
        chunk, document = free[0]
        partner = partner_for(style, free[1:], document)
        chunk_ids = [chunk] + ([partner] if partner is not None else [])
        taken.update(chunk_ids)
        plans.append(TaskPlan(chunk_ids=chunk_ids, style=style))
    return plans


async def start_run(
    session: AsyncSession, count: int, *, document_id: int | None = None
) -> tuple[Job, list[Task]]:
    """Write down a job and the tasks it means to work through."""
    plans = plan_tasks(await candidates(session, document_id=document_id), count)
    job = Job(
        kind="generate",
        options={"count": count, "document_id": document_id, "planned": len(plans)},
        progress=f"0 of {len(plans)} questions",
    )
    session.add(job)
    await session.flush()
    tasks = [
        Task(job_id=job.id, position=position, chunk_ids=plan.chunk_ids, style=plan.style)
        for position, plan in enumerate(plans)
    ]
    session.add_all(tasks)
    await session.flush()
    return job, tasks


async def sources_for(session: AsyncSession, chunk_ids: list[int]) -> list[Source]:
    rows = await session.execute(
        select(Document.title, Chunk)
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.id.in_(chunk_ids))
    )
    found = {chunk.id: source_of(title, chunk) for title, chunk in rows.tuples()}
    return [found[chunk_id] for chunk_id in chunk_ids if chunk_id in found]


def out_of_budget(exc: BaseException, seen: set[int] | None = None) -> bool:
    """Whether a failure is a provider saying it has nothing left for today.

    The cause chain is walked because a fallback chain wraps what each provider raised, and
    it is walked with a guard because those chains can point back at themselves.
    """
    seen = seen if seen is not None else set()
    if id(exc) in seen:
        return False
    seen.add(id(exc))
    if isinstance(exc, QuotaExhausted):
        return True
    if isinstance(exc, BaseExceptionGroup):
        return any(out_of_budget(inner, seen) for inner in exc.exceptions)
    return any(out_of_budget(cause, seen) for cause in (exc.__cause__, exc.__context__) if cause)


async def run_task(
    sessions: async_sessionmaker[AsyncSession],
    task: Task,
    *,
    model: Model,
    checker: Model,
    embedder: Any,
    similarity: float,
) -> tuple[str, dict[str, int]]:
    """Write one question and file it. Returns how it ended and what it cost."""
    async with sessions() as session:
        sources = await sources_for(session, task.chunk_ids)
    if len(sources) != len(task.chunk_ids):
        raise LookupError(f"chunks {task.chunk_ids} are no longer all there")

    generated = await generate_question(model, sources, task.style)
    async with sessions() as session, session.begin():
        checked = await validate(
            session, generated, sources, model=checker, embedder=embedder, similarity=similarity
        )
        topic_id = await topic_for(session, task.chunk_ids)
        question = await store_question(session, generated, checked, sources, topic_id=topic_id)
        await session.execute(
            update(Task)
            .where(Task.id == task.id)
            .values(status="done", question_id=question.id, error=None, finished_at=func.now())
        )
    return ("accepted" if checked.passed else "rejected"), generated.usage


async def run_job(
    sessions: async_sessionmaker[AsyncSession],
    job_id: int,
    *,
    model: Model,
    checker: Model,
    embedder: Any,
    similarity: float,
    report: Reporter | None = None,
) -> Summary:
    """Work through a job's unfinished tasks, committing each one as it goes."""
    say = report or log.info
    summary = Summary()
    async with sessions() as session:
        tasks = list(
            await session.scalars(
                select(Task)
                .where(Task.job_id == job_id, Task.status.in_(("queued", "running")))
                .order_by(Task.position)
            )
        )
    say(f"{len(tasks)} question(s) to write")

    for number, task in enumerate(tasks, start=1):
        await _mark(sessions, task, "running", attempt=True)
        try:
            outcome, usage = await run_task(
                sessions,
                task,
                model=model,
                checker=checker,
                embedder=embedder,
                similarity=similarity,
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"[:2000]
            if out_of_budget(exc):
                # Not the task's fault: leave it queued so a later run picks it up.
                await _mark(sessions, task, "queued", error=message)
                summary.stopped = message
                say(f"stopping, the day's budget is gone: {message}")
                break
            log.exception("task %s failed", task.id)
            await _mark(sessions, task, "failed", error=message)
            summary.failed += 1
            say(f"{number}/{len(tasks)} failed: {message}")
            continue
        summary.written += 1
        setattr(summary, outcome, getattr(summary, outcome) + 1)
        summary.add_usage(usage)
        say(f"{number}/{len(tasks)} {outcome} ({task.style})")

    await _finish(sessions, job_id, summary)
    return summary


async def _mark(
    sessions: async_sessionmaker[AsyncSession],
    task: Task,
    status: str,
    *,
    error: str | None = None,
    attempt: bool = False,
) -> None:
    values: dict[str, Any] = {"status": status, "error": error}
    if attempt:
        values["attempts"] = Task.attempts + 1
    async with sessions() as session, session.begin():
        await session.execute(update(Task).where(Task.id == task.id).values(**values))


async def _finish(
    sessions: async_sessionmaker[AsyncSession], job_id: int, summary: Summary
) -> None:
    async with sessions() as session, session.begin():
        left = await session.scalar(
            select(func.count())
            .select_from(Task)
            .where(Task.job_id == job_id, Task.status == "queued")
        )
        if left:
            # Left queued rather than failed: the work is unfinished, not wrong.
            status = "queued"
        elif summary.failed and not summary.written:
            status = "failed"
        else:
            status = "done"
        await session.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(
                status=status,
                progress=f"{summary.accepted} accepted, {summary.rejected} rejected",
                error=summary.stopped,
                finished_at=None if left else func.now(),
            )
        )

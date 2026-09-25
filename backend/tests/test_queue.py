import asyncio

from sqlalchemy import select
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.db.models import Job
from app.ingest import queue
from app.ingest.storage import StoredFile

OPTIONS = {"ocr": False, "formulas": True}
NOTEBOOK = StoredFile(
    path=f"uploads/{'c' * 64}.ipynb", sha256="c" * 64, source_type="notebook", size=100
)


def test_a_file_is_recorded_once_and_its_active_job_reused(sessions) -> None:
    async def scenario():
        async with sessions() as session:
            first = await queue.add_file(session, NOTEBOOK, "Lesson 1.ipynb", OPTIONS)
            again = await queue.add_file(session, NOTEBOOK, "copy.ipynb", OPTIONS)
            await session.commit()
        return first, again

    first, again = asyncio.run(scenario())

    assert first.message == "queued"
    assert (first.document.title, first.document.filename) == ("Lesson 1", "Lesson 1.ipynb")
    assert (first.document.source_type, first.document.path) == ("notebook", NOTEBOOK.path)
    assert (first.job.status, first.job.options) == ("queued", OPTIONS)
    assert again.message == "already queued"
    assert (again.document.id, again.job.id) == (first.document.id, first.job.id)


def test_a_ready_document_is_ingested_again_only_when_forced(sessions) -> None:
    async def scenario():
        async with sessions() as session:
            added = await queue.add_file(session, NOTEBOOK, "lesson.ipynb", OPTIONS)
            added.job.status = "done"
            added.document.status = "ready"
            await session.flush()
            skipped = await queue.add_file(session, NOTEBOOK, "lesson.ipynb", OPTIONS)
            forced = await queue.add_file(session, NOTEBOOK, "lesson.ipynb", OPTIONS, force=True)
        return added, skipped, forced

    added, skipped, forced = asyncio.run(scenario())

    assert skipped.job is None
    assert skipped.message.startswith("already ingested")
    assert forced.message == "queued"
    assert forced.job.id != added.job.id


def test_a_different_arxiv_version_is_ingested_again(sessions) -> None:
    async def scenario():
        async with sessions() as session:
            added = await queue.add_arxiv(session, "1706.03762", None, OPTIONS)
            added.job.status = "done"
            added.document.status = "ready"
            added.document.arxiv_version = "v7"
            await session.flush()
            same = await queue.add_arxiv(session, "1706.03762", "v7", OPTIONS)
            older = await queue.add_arxiv(session, "1706.03762", "v5", OPTIONS)
        return added, same, older

    added, same, older = asyncio.run(scenario())

    assert (added.document.title, added.document.source_type) == ("arXiv:1706.03762", "arxiv")
    assert added.job.options == {**OPTIONS, "version": None}
    assert same.job is None
    assert older.document.id == added.document.id
    assert older.job.options == {**OPTIONS, "version": "v5"}


def test_jobs_are_claimed_oldest_first(sessions) -> None:
    async def scenario():
        async with sessions() as session:
            first = await queue.add_arxiv(session, "1706.03762", None, OPTIONS)
            second = await queue.add_arxiv(session, "2510.10824", None, OPTIONS)
            await session.commit()
        async with sessions() as session:
            claimed = [await queue.claim_next_job(session) for _ in range(3)]
            jobs = list(await session.scalars(select(Job).order_by(Job.id)))
        return [first.job.id, second.job.id], claimed, jobs

    queued, claimed, jobs = asyncio.run(scenario())

    assert claimed == [*queued, None]
    for job in jobs:
        assert (job.status, job.progress, job.attempts) == ("running", "starting", 1)
        assert job.started_at is not None


def test_each_kind_of_job_is_claimed_on_its_own(sessions) -> None:
    async def scenario():
        async with sessions() as session, session.begin():
            ingest = (await queue.add_arxiv(session, "1706.03762", None, OPTIONS)).job
            topics = Job(kind="topics")
            generate = Job(kind="generate", options={"count": 3, "planned": 3})
            session.add_all([topics, generate])
        async with sessions() as session:
            claimed = [
                await queue.claim_next_job(session, kind)
                for kind in ("topics", "topics", "generate", "ingest")
            ]
        return [topics.id, None, generate.id, ingest.id], claimed

    expected, claimed = asyncio.run(scenario())

    assert claimed == expected


def test_a_job_locked_by_another_worker_is_skipped(sessions) -> None:
    async def scenario():
        async with sessions() as session:
            first = await queue.add_arxiv(session, "1706.03762", None, OPTIONS)
            second = await queue.add_arxiv(session, "2510.10824", None, OPTIONS)
            await session.commit()
        async with sessions() as holder, sessions() as claimer:
            # Another worker has locked the oldest job and not committed yet.
            await holder.execute(select(Job).where(Job.id == first.job.id).with_for_update())
            claimed = await queue.claim_next_job(claimer)
            await holder.rollback()
        return second.job.id, claimed

    expected, claimed = asyncio.run(scenario())

    assert claimed == expected


def test_interrupted_jobs_are_queued_again(sessions) -> None:
    async def scenario():
        async with sessions() as session:
            added = await queue.add_arxiv(session, "1706.03762", None, OPTIONS)
            await session.commit()
            await queue.claim_next_job(session)
            requeued = await queue.requeue_interrupted(session)
            job = await session.get_one(Job, added.job.id, populate_existing=True)
        return requeued, job

    requeued, job = asyncio.run(scenario())

    assert requeued == 1
    assert (job.status, job.progress) == ("queued", "queued again after an interruption")


def test_only_one_process_may_ingest_at_a_time(engine) -> None:
    async def scenario() -> list[bool]:
        async with queue.ingest_lock(engine) as first, queue.ingest_lock(engine) as second:
            held = [first, second]
        async with queue.ingest_lock(engine) as after_release:
            return [*held, after_release]

    assert asyncio.run(scenario()) == [True, False, True]


def test_a_worker_is_seen_by_the_lock_it_holds(engine, sessions, database_url) -> None:
    # A worker on another database of the same server, such as the development one
    elsewhere = create_async_engine(
        make_url(database_url).set(database="postgres"), poolclass=NullPool
    )

    async def running() -> bool:
        async with sessions() as session:
            return await queue.worker_running(session)

    async def scenario() -> list[bool]:
        seen = [await running()]
        async with queue.ingest_lock(elsewhere) as other:
            seen += [other, await running()]
        async with queue.ingest_lock(engine) as ours:
            seen += [ours, await running()]
        seen.append(await running())
        await elsewhere.dispose()
        return seen

    assert asyncio.run(scenario()) == [False, True, False, True, True, False]

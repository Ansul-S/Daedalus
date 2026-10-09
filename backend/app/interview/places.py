"""Where LangGraph keeps an interview's place between requests (`app.interview.graph`): the
saver's rows for the thread "interview-<id>". They are deleted once the interview is over, and
with the practice it belongs to, in the caller's transaction."""

from collections.abc import Sequence

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import CHECKPOINT_BLOBS, CHECKPOINT_WRITES, CHECKPOINTS


def thread(interview_id: int) -> str:
    return f"interview-{interview_id}"


async def forget(session: AsyncSession, interview_ids: Sequence[int]) -> None:
    """Delete the saved places of these interviews."""
    if not interview_ids:
        return
    threads = [thread(interview_id) for interview_id in interview_ids]
    for table in (CHECKPOINT_WRITES, CHECKPOINT_BLOBS, CHECKPOINTS):
        await session.execute(delete(table).where(table.c.thread_id.in_(threads)))

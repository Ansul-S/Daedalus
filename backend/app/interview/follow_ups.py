"""Follow-up questions: one written for an answer, about the gap in it, from the passages its
question was written from.

A follow-up is graded the way its question was, against key points that each rest on a
sentence of those passages (`app.grading.grader`), so it has to come with them. A writer that
can't produce one that holds up gives None, and the interview moves on without it.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.interview.routing import Gap
from app.questions.generation import Source


@dataclass(frozen=True)
class FollowUp:
    text: str
    # What an answer has to cover, as a library question's key points have it:
    # [{"text", "weight", "evidence_quote", "chunk_id"}]
    key_points: list[dict[str, Any]]
    # The passages it rests on, in the order they are shown to the grader
    chunk_ids: list[int]
    model: str
    prompt_version: str
    usage: dict[str, int] = field(default_factory=dict)


class Writer(Protocol):
    async def __call__(
        self, question: str, sources: Sequence[Source], gap: Gap, answer: str
    ) -> FollowUp | None: ...

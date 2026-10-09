"""What a follow-up aims at, decided by rules on the grade rather than by a model.

An interviewer follows up where an answer went wrong or fell short. A claim a passage
contradicts comes first: the candidate holds something the source says is not so, and the
follow-up asks about what the passage says instead. Otherwise the follow-up asks about the
weightiest key point the answer missed or only gestured at, a missed one before a partial one
of the same weight. An answer that covered what the passages ask, with nothing contradicted,
gets none: a deeper question would have to go past the key points, where nothing checks it.
"""

from dataclasses import dataclass
from typing import Any, Literal

from app.db.models import Grade

# A score at or above this, with nothing contradicted, leaves nothing to follow up
COVERED_AT = 0.85

# Why no follow-up came, as the interview says it
NOT_GRADED = "The answer could not be graded, so there was nothing to follow up."
COVERED = "The answer covered what the passages ask."
NO_GRADE_LEFT = "No grade was left today for a follow-up."
NO_WRITER = "Follow-ups can't be written here."
NOT_WRITTEN = "No follow-up held up to the checks."

Kind = Literal["contradicted", "missing", "partial"]


@dataclass(frozen=True)
class Gap:
    kind: Kind
    # The key point missed or only partly made, as the question has it, with its id
    point: dict[str, Any] | None = None
    # The claim a passage contradicts: {"claim", "why", "chunk_id"}
    claim: dict[str, Any] | None = None

    @property
    def chunk_id(self) -> int | None:
        """The passage the follow-up should rest on, when the gap names one."""
        found = self.claim if self.claim is not None else self.point
        return found.get("chunk_id") if found else None

    @property
    def text(self) -> str:
        """The gap in a sentence: the claim, or the key point."""
        return self.claim["claim"] if self.claim is not None else (self.point or {})["text"]

    def as_aim(self) -> dict[str, Any]:
        """As a follow-up records what it aims at."""
        return {"kind": self.kind, "text": self.text, "chunk_id": self.chunk_id}


def aim(grade: Grade | None, key_points: list[dict[str, Any]]) -> Gap | str:
    """The gap a follow-up should ask about, or why there is none to ask about."""
    if grade is None or grade.status != "graded" or grade.score is None:
        return NOT_GRADED
    for claim in grade.claims:
        if claim["verdict"] == "contradicted" and claim.get("chunk_id") is not None:
            return Gap(
                "contradicted", claim={key: claim[key] for key in ("claim", "why", "chunk_id")}
            )
    if grade.score >= COVERED_AT:
        return COVERED
    short = [
        (label, point)
        for label, point in zip(grade.key_points, key_points, strict=False)
        if label["status"] in ("missing", "partial")
    ]
    if not short:
        return COVERED
    # The first of the weightiest, a missed point before a partial one
    label, point = max(short, key=lambda pair: (pair[1]["weight"], pair[0]["status"] == "missing"))
    return Gap(label["status"], point=point | {"id": label["id"]})

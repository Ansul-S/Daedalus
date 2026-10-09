"""What an interview comes to, worked out from its grades when it is asked for.

Every question's own verdict is already on its turn; the report adds up what the interview
showed: how many questions were answered and how well, how the follow-ups went, and the topics
to go back to. Nothing in it is a model's: it is counted from the grades.
"""

from dataclasses import dataclass
from datetime import date

# A main answer or a follow-up below this sends its topic to the review list
REVIEW_BELOW = 0.6
# A follow-up at or above this made up what the answer missed
RECOVERED_AT = 0.6


@dataclass(frozen=True)
class Round:
    """One library question of an interview, as the report sees it."""

    question_id: int
    topic_id: int | None
    answered: bool
    # The answer's score and its follow-up's; None when there is no grade to read
    score: float | None
    follow_up_score: float | None
    # The rating the answer earned for the schedule, and the day the question comes back
    rating: str | None
    due: date | None

    @property
    def recovered(self) -> bool | None:
        """Whether the follow-up made up for the gap; None without a graded follow-up."""
        return None if self.follow_up_score is None else self.follow_up_score >= RECOVERED_AT

    @property
    def to_review(self) -> bool:
        return (self.score is not None and self.score < REVIEW_BELOW) or self.recovered is False


@dataclass(frozen=True)
class Report:
    answered: int
    # Over the answers that were graded; None when none was
    mean_score: float | None
    # Follow-ups answered and graded, and those that made up for the gap
    follow_ups: int
    recovered: int
    # The topics to go back to, in the order they came up
    review: list[int | None]


def report(rounds: list[Round]) -> Report:
    scores = [round_.score for round_ in rounds if round_.score is not None]
    followed = [round_ for round_ in rounds if round_.recovered is not None]
    review: list[int | None] = []
    for round_ in rounds:
        if round_.to_review and round_.topic_id not in review:
            review.append(round_.topic_id)
    return Report(
        answered=sum(round_.answered for round_ in rounds),
        mean_score=round(sum(scores) / len(scores), 4) if scores else None,
        follow_ups=len(followed),
        recovered=sum(bool(round_.recovered) for round_ in followed),
        review=review,
    )

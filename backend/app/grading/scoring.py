"""Turning a grader's labels into a score.

The grader never gives a number. It says whether each key point is covered, partly covered or
missing, and whether each claim in the answer agrees with the sources; the score follows
from those labels and from the weights the key points were written with. The same labels
therefore always give the same score, and the score can be explained point by point.

A claim the sources contradict costs a fixed amount, since a confident mistake is worse in
an interview than a gap. A claim the sources simply do not cover costs nothing: it may well
be true.
"""

from collections.abc import Sequence
from dataclasses import dataclass

CREDIT = {"covered": 1.0, "partial": 0.5, "missing": 0.0}
# Taken off per contradicted claim
CONTRADICTION_PENALTY = 0.15


@dataclass(frozen=True)
class Score:
    # Share of the key points' weight the answer covers, 0 to 1
    coverage: float
    contradicted: int
    # Coverage less the penalty, never below 0
    score: float


def score(weights: Sequence[int], statuses: Sequence[str], contradicted: int) -> Score:
    """The score for an answer, from each key point's weight and label, in the same order."""
    if len(weights) != len(statuses):
        raise ValueError(f"{len(statuses)} labels for {len(weights)} key points")
    total = sum(weights)
    if total <= 0:
        raise ValueError("the key points carry no weight")
    coverage = sum(w * CREDIT[s] for w, s in zip(weights, statuses, strict=True)) / total
    return Score(
        coverage=round(coverage, 4),
        contradicted=contradicted,
        score=round(max(0.0, coverage - CONTRADICTION_PENALTY * contradicted), 4),
    )

"""Agreement between the human labels and an automated judge.

Pure functions: pairs of labels go in, a number comes out. Nothing here touches
the database or a model, so every value can be checked against an example worked
out by hand, and the tests need no PostgreSQL.

A pair is ``(human, judge)`` for one question on one rubric. The human label is
the reference, not because it is correct — it is one annotator's judgement, made
once — but because it is the thing the judge is being measured against.

Four decisions shape this module.

**Raw agreement and a chance-corrected figure are always reported together.**
Raw agreement alone is inflated by a skewed scale: if 60% of questions carry the
same label, a judge that always guesses that label agrees 60% of the time while
knowing nothing. Cohen's kappa subtracts the agreement expected from the two
raters' marginals alone.

**Kappa is depressed by skewed marginals, and that is not a defect in the
judge.** Where one category dominates, expected agreement is high, so the
remaining room for improvement is small and kappa falls even when raw agreement
is high. This is the well-known kappa paradox. Every report therefore carries
the marginals and the confusion counts, so a low kappa beside a high raw rate
can be read for what it is rather than quoted as a verdict.

**Disagreement weights are linear, and only on an ordinal scale.** Groundedness
0/1/2 and difficulty easy/medium/hard are ordered, so a one-step disagreement is
a smaller error than a two-step one and unweighted kappa would discard that.
Linear weights make the smaller assumption: distance is proportional to steps
apart. The common quadratic convention would treat a two-step disagreement as
four times a one-step one, which the scales do not establish and which happens to
produce a larger number. ``unusable`` sits on no ordinal line — a question too
incoherent to place is not "more than hard" — so any category set containing it
is nominal and takes unweighted kappa.

**A judge label off its own scale is not silently dropped.** The protocol stores
judge output exactly as emitted, so off-scale values reach this module and every
function that consumes pairs requires an explicit policy with no default. Neither
treatment is a project-wide answer: excluding measures the judge where it did the
task, counting it as disagreement charges it for failing to. Both are reported.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

#: One question's labels on one rubric: (human, judge).
Pair = tuple[str, str]

#: The ordered part of each rubric's scale. Mirrors the CHECK constraints added
#: by migration 007, restated here so this module imports no database code.
ORDINAL_SCALES: dict[str, tuple[str, ...]] = {
    "groundedness": ("0", "1", "2"),
    "relevance": ("0", "1", "2"),
    "difficulty": ("easy", "medium", "hard"),
}

#: The difficulty label that sits outside the ordering. A question too incoherent
#: to carry a difficulty is not a fourth degree of difficulty.
UNUSABLE = "unusable"

#: How a pair whose human label is ``unusable`` is treated. Difficulty only.
#:
#: ``"exclude"``  drops the pair, mirroring what the rubrics do for the
#:                generator's difficulty match, and leaves a purely ordinal scale.
#: ``"include"``  keeps ``unusable`` as a fourth, unordered category, which asks
#:                whether the judge can also recognise an incoherent question.
UnusablePolicy = Literal["exclude", "include"]

#: How a judge label outside the rubric's scale is treated.
#:
#: ``"exclude"``   drops the pair and reports how many were dropped.
#: ``"disagree"``  keeps the pair as a non-match, charging the judge for
#:                 answering off its own scale.
OffScalePolicy = Literal["exclude", "disagree"]

#: Both supported treatments of each. Neither is a default; the caller chooses.
UNUSABLE_POLICIES: tuple[UnusablePolicy, ...] = ("exclude", "include")
OFF_SCALE_POLICIES: tuple[OffScalePolicy, ...] = ("exclude", "disagree")

#: Bootstrap resamples used for a confidence interval, and the seed that makes a
#: reported interval reproducible. Restated rather than imported from the
#: retrieval harness, which pulls in psycopg.
DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20260909


def categories(rubric: str, unusable: UnusablePolicy) -> tuple[str, ...]:
    """Return the label set for a rubric under an ``unusable`` policy.

    Only difficulty has an ``unusable`` label, so the policy is inert elsewhere
    rather than an error: it lets one caller loop over all three rubrics under
    the same policy without special-casing.
    """
    _check_rubric(rubric)
    scale = ORDINAL_SCALES[rubric]
    if rubric == "difficulty" and unusable == "include":
        return (*scale, UNUSABLE)
    return scale


def is_ordinal(rubric: str, unusable: UnusablePolicy) -> bool:
    """Whether the label set is ordered, and so whether weights are meaningful."""
    return UNUSABLE not in categories(rubric, unusable)


def prepare(
    pairs: Sequence[Pair],
    rubric: str,
    unusable: UnusablePolicy,
    off_scale: OffScalePolicy,
) -> tuple[list[Pair], int, int]:
    """Apply both policies, returning the usable pairs and what they cost.

    Returns ``(kept, unusable_dropped, off_scale_seen)``. A human label that is
    not on the rubric's scale at all is a defect in the stored labels rather than
    a policy question, and raises.
    """
    _check_rubric(rubric)
    _check_policies(unusable, off_scale)
    valid = categories(rubric, unusable)
    human_valid = categories(rubric, "include")

    kept: list[Pair] = []
    unusable_dropped = 0
    off_scale_seen = 0
    for human, judge in pairs:
        if human not in human_valid:
            raise ValueError(f"human label {human!r} is not on the {rubric} scale")
        if human == UNUSABLE and unusable == "exclude":
            unusable_dropped += 1
            continue
        if judge not in valid:
            off_scale_seen += 1
            if off_scale == "exclude":
                continue
        kept.append((human, judge))
    return kept, unusable_dropped, off_scale_seen


def confusion(pairs: Sequence[Pair]) -> dict[Pair, int]:
    """Count each (human, judge) combination that occurs."""
    counts: dict[Pair, int] = {}
    for pair in pairs:
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def marginals(pairs: Sequence[Pair]) -> tuple[dict[str, int], dict[str, int]]:
    """Return each rater's own label distribution, human first."""
    human: dict[str, int] = {}
    judge: dict[str, int] = {}
    for h, j in pairs:
        human[h] = human.get(h, 0) + 1
        judge[j] = judge.get(j, 0) + 1
    return human, judge


def raw_agreement(pairs: Sequence[Pair]) -> float | None:
    """Share of pairs where the two labels are identical.

    Returns None on no pairs, where a rate would be a division by zero dressed
    up as a result.
    """
    if not pairs:
        return None
    return sum(1 for human, judge in pairs if human == judge) / len(pairs)


def cohen_kappa(
    pairs: Sequence[Pair],
    scale: Sequence[str],
    weights: Literal["none", "linear"],
) -> float | None:
    """Cohen's kappa over a declared label set, unweighted or linearly weighted.

    ``scale`` is the full set of labels, in the rubric's own order. It is passed
    explicitly rather than inferred from the pairs for two reasons: it is what
    makes an off-scale label an error instead of a new category, and under linear
    weights it fixes which labels are adjacent. Adding a category neither rater
    used does not move the result — an unused category contributes nothing to
    expected disagreement, and the step normalisation cancels — so the scale
    matters for what it rejects and how it orders, not as a correction term.

    Returns None when expected disagreement is zero — both raters used exactly
    one category, and there is no chance-corrected question to answer.
    """
    if not pairs:
        return None
    if weights not in ("none", "linear"):
        raise ValueError(f"unknown weights {weights!r}")

    off = {label for pair in pairs for label in pair} - set(scale)
    if off:
        raise ValueError(f"labels outside the declared scale: {sorted(off)}")

    n = len(pairs)
    counts = confusion(pairs)
    human, judge = marginals(pairs)
    steps = len(scale) - 1

    def cost(a: str, b: str) -> float:
        if weights == "none":
            return 0.0 if a == b else 1.0
        return abs(scale.index(a) - scale.index(b)) / steps

    observed = sum(cost(h, j) * count for (h, j), count in counts.items()) / n
    expected = sum(
        cost(h, j) * human.get(h, 0) * judge.get(j, 0) / (n * n)
        for h in scale
        for j in scale
    )
    if expected == 0:
        return None
    return 1.0 - observed / expected


@dataclass(frozen=True)
class Agreement:
    """One rubric's agreement under one pair of policies.

    ``n`` is the number of pairs the figures were computed on, which is the only
    denominator any of them refer to. ``unusable_dropped`` and ``off_scale`` say
    how far it sits below the number of questions labelled, so a rate is never
    reported without what it was computed on being visible.
    """

    rubric: str
    unusable: UnusablePolicy
    off_scale_policy: OffScalePolicy
    n: int
    unusable_dropped: int
    off_scale: int
    raw: float | None
    kappa: float | None
    weights: Literal["none", "linear"]
    human_marginals: dict[str, int]
    judge_marginals: dict[str, int]
    confusion: dict[Pair, int]


def measure(
    pairs: Sequence[Pair],
    rubric: str,
    unusable: UnusablePolicy,
    off_scale: OffScalePolicy,
) -> Agreement:
    """Compute one rubric's agreement under both explicit policies.

    Weighting follows the label set rather than the caller: an ordinal set takes
    linear weights, and a set containing ``unusable`` takes none, because a
    distance to an unordered category would be invented rather than measured.
    """
    kept, dropped, off = prepare(pairs, rubric, unusable, off_scale)
    scale = categories(rubric, unusable)
    weights: Literal["none", "linear"] = (
        "linear" if is_ordinal(rubric, unusable) else "none"
    )

    # Off-scale judge labels kept as disagreements sit outside the declared
    # scale, so a chance-corrected figure would need an expected rate for a
    # category the scale does not contain. Raw agreement still means what it
    # says, and is reported without a kappa beside it.
    kappa = (
        None if off and off_scale == "disagree" else cohen_kappa(kept, scale, weights)
    )
    human, judge = marginals(kept)
    return Agreement(
        rubric=rubric,
        unusable=unusable,
        off_scale_policy=off_scale,
        n=len(kept),
        unusable_dropped=dropped,
        off_scale=off,
        raw=raw_agreement(kept),
        kappa=kappa,
        weights=weights,
        human_marginals=human,
        judge_marginals=judge,
        confusion=confusion(kept),
    )


def bootstrap_ci(
    pairs: Sequence[Pair],
    estimator: Callable[[Sequence[Pair]], float | None],
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = 0.95,
) -> tuple[float, float] | None:
    """Percentile bootstrap interval for a statistic over questions.

    Questions are the unit of resampling because they are what was sampled: 300
    sections drawn by hash, of which 228 produced a question. The interval says
    how much of the estimate that sample supports, and nothing about whether the
    human label was right.

    Resamples on which the estimator is undefined are discarded rather than
    counted as zero, and the interval is None when fewer than two survive.
    """
    if len(pairs) < 2:
        return None
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(resamples):
        drawn = [rng.choice(pairs) for _ in pairs]
        value = estimator(drawn)
        if value is not None:
            values.append(value)
    if len(values) < 2:
        return None
    values.sort()
    tail = (1.0 - confidence) / 2.0
    low = values[int(tail * len(values))]
    high = values[min(int((1.0 - tail) * len(values)), len(values) - 1)]
    return (low, high)


def _check_rubric(rubric: str) -> None:
    if rubric not in ORDINAL_SCALES:
        raise ValueError(f"unknown rubric {rubric!r}")


def _check_policies(unusable: UnusablePolicy, off_scale: OffScalePolicy) -> None:
    if unusable not in UNUSABLE_POLICIES:
        raise ValueError(f"unknown unusable policy {unusable!r}")
    if off_scale not in OFF_SCALE_POLICIES:
        raise ValueError(f"unknown off-scale policy {off_scale!r}")

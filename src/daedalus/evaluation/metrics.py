"""Ranking metrics over the hand-labelled reference set.

Pure functions: a ranking and one query's judgements go in, a number comes out.
Nothing here touches the database, an embedder or a model, so every value can be
checked against an example worked out by hand.

Four properties of the reference set shape this module.

**Judgements are keyed by (doc_id, ordinal), never by chunk id.** The judgements
table carries no foreign key into chunks, so the reference set survives a
re-ingest that renumbers them. The metrics join on the same pair, and stay valid
across a re-ingest for the same reason the schema does.

**Relevance is graded 0, 1, 2, and the grades mean different things.** Grade 2 is
sufficient to answer the question on its own; grade 1 contributes useful evidence
but does not suffice; grade 0 does not help. NDCG consumes those grades directly.
The binary metrics need a cut, and both "useful evidence" (>= 1) and "sufficient
evidence" (= 2) are meaningful cuts, so the threshold is always an explicit
argument and never a default.

**Gain is linear**: grades 0, 1 and 2 give gains 0, 1 and 2. The grading policy
establishes that a grade 2 answers the question and a grade 1 does not, but it
does not establish that a grade 2 is worth disproportionately more than a grade 1.
Linear gain makes the smaller assumption. The common exponential convention,
2 ** g - 1, would weigh a grade 2 three times a grade 1 on no evidence from the
policy that graded them.

**An unjudged chunk is not a grade 0 chunk.** It was never assessed. Every
function that consumes a ranking therefore requires an explicit ``unjudged``
policy with no default, so no number can be produced without stating how
unjudged candidates were treated. Neither policy is a project-wide default: for a
retriever that contributed to the judged pool the choice is conventional, but for
one that did not it has to be made from measured pool overlap, which is what
``unjudged_count`` is for.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import log2
from typing import Literal

#: A chunk as the reference set identifies it: (doc_id, ordinal).
ChunkRef = tuple[str, int]

#: One query's judgements. Absence from this mapping means unjudged, not zero.
Grades = Mapping[ChunkRef, int]

#: How a retrieved chunk carrying no judgement is treated.
#:
#: ``"zero"``  counts it as grade 0. The standard pooled-evaluation convention,
#:             and the pessimistic one: it charges a retriever for surfacing
#:             chunks nobody looked at.
#: ``"skip"``  removes it from the ranking, closing the gap, so the metric is
#:             computed over the judged subset only.
UnjudgedPolicy = Literal["zero", "skip"]

#: Both supported treatments. Neither is a default; the caller must choose.
UNJUDGED_POLICIES: tuple[UnjudgedPolicy, ...] = ("zero", "skip")

#: Valid relevance grades. Mirrors the CHECK constraint on ``judgements`` and
#: ``storage.queries.GRADES``, restated here so this module imports no database
#: code and its tests need no PostgreSQL.
GRADES = (0, 1, 2)

#: Thresholds at which a graded judgement counts as relevant for a binary
#: metric: 1 is "useful evidence", 2 is "sufficient evidence".
THRESHOLDS = (1, 2)


def precision_at_k(
    ranking: Sequence[ChunkRef],
    grades: Grades,
    k: int,
    threshold: int,
    unjudged: UnjudgedPolicy,
) -> float:
    """Fraction of the top k that is relevant at the given threshold.

    The denominator is the number of positions actually available, min(k, len),
    not k itself. A retriever that returns fewer than k chunks is not charged
    for positions holding nothing, and under ``"skip"`` it is not charged for
    the unjudged positions the policy just removed -- which is the entire point
    of that policy. This diverges from trec_eval, which always divides by k. The
    two agree whenever the ranking is at least k long, which is the ordinary
    case here since the retrievers are called with a limit of k or more.

    An empty ranking scores 0.0: nothing relevant was retrieved.
    """
    _check_k(k)
    _check_threshold(threshold)
    graded = _graded(ranking, grades, unjudged)[:k]
    if not graded:
        return 0.0
    return sum(1 for grade in graded if grade >= threshold) / len(graded)


def recall_at_k(
    ranking: Sequence[ChunkRef],
    grades: Grades,
    k: int,
    threshold: int,
    unjudged: UnjudgedPolicy,
) -> float | None:
    """Fraction of the query's relevant chunks that appear in the top k.

    Returns None when the query has no relevant chunk at this threshold, where
    recall is undefined rather than zero. Averaging such a query in as 0.0 would
    understate every system, so the caller has to exclude it deliberately.

    The denominator is the query's whole judged relevant set and is unaffected
    by the unjudged policy. Relevance is only ever established by a judgement,
    so every relevant chunk is judged by construction; the policy governs what
    the ranking contributes to the numerator, not what exists to be found.
    """
    _check_k(k)
    _check_threshold(threshold)
    graded = _graded(ranking, grades, unjudged)
    relevant = sum(1 for grade in grades.values() if grade >= threshold)
    if relevant == 0:
        return None
    return sum(1 for grade in graded[:k] if grade >= threshold) / relevant


def reciprocal_rank(
    ranking: Sequence[ChunkRef],
    grades: Grades,
    threshold: int,
    unjudged: UnjudgedPolicy,
) -> float:
    """Reciprocal of the rank of the first relevant chunk, or 0.0 if there is none.

    Computed over the whole ranking with no cut-off, so the caller sets the
    horizon by choosing how many chunks to retrieve. Under ``"skip"`` the ranks
    close up, so a chunk preceded only by unjudged candidates is rank 1.
    """
    _check_threshold(threshold)
    graded = _graded(ranking, grades, unjudged)
    for position, grade in enumerate(graded, start=1):
        if grade >= threshold:
            return 1.0 / position
    return 0.0


def ndcg_at_k(
    ranking: Sequence[ChunkRef],
    grades: Grades,
    k: int,
    unjudged: UnjudgedPolicy,
) -> float | None:
    """Discounted cumulative gain at k, over the ideal ordering's.

    Gain is the grade itself (see the module docstring). The discount is
    1 / log2(i + 1) with i counted from 1, so rank 1 is undiscounted.

    The ideal ordering is built from the query's complete judged set, sorted by
    grade and truncated at k -- not from whatever the ranking happened to
    retrieve. Building it from the retrieved chunks is the classic error: it
    scores any ranking that found one relevant chunk and put it first as
    perfect.

    Returns None when the ideal gain is zero, meaning every judged chunk for
    this query is grade 0, where NDCG is undefined. Because gains are graded,
    a query needs only a single grade 1 to be measurable, so no threshold-driven
    exclusion applies to this metric.
    """
    _check_k(k)
    graded = _graded(ranking, grades, unjudged)
    ideal = sorted(grades.values(), reverse=True)[:k]
    ideal_gain = _dcg(ideal)
    if ideal_gain == 0.0:
        return None
    return _dcg(graded[:k]) / ideal_gain


def unjudged_count(ranking: Sequence[ChunkRef], grades: Grades, k: int) -> int:
    """How many of the top k retrieved chunks carry no judgement.

    Reported beside every metric so a reader can see how much of a ranking the
    reference set actually covers. For a retriever that did not contribute to
    the pool this is the measurement that says whether its evaluation is
    interpretable at all: a challenger whose top k is largely unjudged is being
    scored on chunks nobody looked at, not on chunks known to be irrelevant.

    Counted on the raw ranking, before any unjudged policy is applied, since
    the point is to measure what the policy would be acting on.
    """
    _check_k(k)
    _check_ranking(ranking)
    return sum(1 for ref in ranking[:k] if ref not in grades)


def _dcg(gains: Sequence[int]) -> float:
    """Discounted cumulative gain: linear gain over a log2(i + 1) discount."""
    return sum(
        (gain / log2(position + 1) for position, gain in enumerate(gains, start=1)),
        0.0,
    )


def _graded(
    ranking: Sequence[ChunkRef], grades: Grades, unjudged: UnjudgedPolicy
) -> list[int]:
    """Return the ranking as a list of grades, with the unjudged policy applied."""
    if unjudged not in UNJUDGED_POLICIES:
        raise ValueError(
            f"unjudged must be one of {UNJUDGED_POLICIES}, got {unjudged!r}"
        )
    _check_ranking(ranking)
    _check_grades(grades)
    if unjudged == "skip":
        return [grades[ref] for ref in ranking if ref in grades]
    return [grades.get(ref, 0) for ref in ranking]


def _check_ranking(ranking: Sequence[ChunkRef]) -> None:
    """Reject a ranking that lists the same chunk twice.

    A retriever returning one chunk at two ranks is a bug, and absorbing it
    silently would let it inflate precision and recall.
    """
    seen: set[ChunkRef] = set()
    for ref in ranking:
        if ref in seen:
            raise ValueError(f"ranking contains {ref!r} more than once")
        seen.add(ref)


def _check_grades(grades: Grades) -> None:
    """Reject grades outside the labelling scale."""
    unknown = sorted({grade for grade in grades.values() if grade not in GRADES})
    if unknown:
        raise ValueError(f"grades must be one of {GRADES}, got {unknown}")


def _check_k(k: int) -> None:
    """Reject a non-positive cut-off, which is a caller error rather than data."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")


def _check_threshold(threshold: int) -> None:
    """Reject a relevance threshold the grading scale does not support."""
    if threshold not in THRESHOLDS:
        raise ValueError(f"threshold must be one of {THRESHOLDS}, got {threshold!r}")

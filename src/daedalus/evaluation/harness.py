"""Running a retriever over the reference set and scoring what it returns.

The split from `metrics.py` is deliberate and load-bearing. Everything in that
module is a pure function of a ranking and a set of grades; everything here
touches the database, the embedding service, or the clock. Keeping them apart is
what lets the arithmetic be tested against examples worked out by hand, with no
PostgreSQL and no model.

A retriever is a plain callable from query text to a ranked list of chunk
references. Nothing more is required of it, so the four being compared -- two
that built the judged pool and two that did not -- are the same kind of object
here and are scored by identical code.

Aggregation is macro-averaged: the mean of the per-query scores, not a pooled
count. That is what trec_eval reports and it weights every query equally
regardless of how many relevant chunks it happens to have. A query whose metric
is undefined is excluded from that metric's mean rather than counted as zero, so
every aggregate carries the number of queries actually behind it.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import psycopg

from daedalus.evaluation.metrics import (
    ChunkRef,
    Grades,
    UnjudgedPolicy,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    unjudged_count,
)

Connection = psycopg.Connection[tuple[object, ...]]

#: A retriever: query text in, ranked chunk references out.
Retriever = Callable[[str], list[ChunkRef]]

#: One query's scores, keyed by metric name. None means undefined, never zero.
Scores = dict[str, float | None]

#: Bootstrap resamples used for a confidence interval.
DEFAULT_RESAMPLES = 10_000

#: Seed for the bootstrap, so a reported interval can be reproduced exactly.
DEFAULT_SEED = 20260906


def load_queries(connection: Connection) -> list[tuple[int, str, str]]:
    """Return (id, text, source) for every query, lowest id first."""
    with connection.cursor() as cursor:
        cursor.execute("SELECT id, text, source FROM queries ORDER BY id")
        return [
            (cast("int", row[0]), cast("str", row[1]), cast("str", row[2]))
            for row in cursor.fetchall()
        ]


def load_grades(connection: Connection) -> dict[int, Grades]:
    """Return every query's judgements, keyed by query id then by chunk.

    Absence from a query's mapping means unjudged. It does not mean grade 0, and
    the metrics require the caller to say which of those two it is.
    """
    grades: dict[int, dict[ChunkRef, int]] = {}
    with connection.cursor() as cursor:
        cursor.execute("SELECT query_id, doc_id, ordinal, grade FROM judgements")
        for row in cursor.fetchall():
            query_id = cast("int", row[0])
            ref = (cast("str", row[1]), cast("int", row[2]))
            grades.setdefault(query_id, {})[ref] = cast("int", row[3])
    return {query_id: cast("Grades", g) for query_id, g in grades.items()}


def run_retriever(
    queries: Sequence[tuple[int, str, str]], retriever: Retriever
) -> dict[int, list[ChunkRef]]:
    """Run one retriever over every query, returning its ranking per query."""
    return {query_id: retriever(text) for query_id, text, _ in queries}


def score_ranking(
    ranking: Sequence[ChunkRef],
    grades: Grades,
    ks: Sequence[int],
    unjudged: UnjudgedPolicy,
    thresholds: Sequence[int] = (1, 2),
) -> Scores:
    """Score one ranking against one query's judgements.

    Metric names are `ndcg@K`, `p@K_tT`, `r@K_tT` and `rr_tT`, plus
    `unjudged@K`, which records how much of the ranking the reference set covers
    at all. That last one is not a quality measure: it is the number that says
    whether the others can be interpreted.
    """
    scores: Scores = {}
    for k in ks:
        scores[f"ndcg@{k}"] = ndcg_at_k(ranking, grades, k, unjudged)
        scores[f"unjudged@{k}"] = float(unjudged_count(ranking, grades, k))
        for threshold in thresholds:
            scores[f"p@{k}_t{threshold}"] = precision_at_k(
                ranking, grades, k, threshold, unjudged
            )
            scores[f"r@{k}_t{threshold}"] = recall_at_k(
                ranking, grades, k, threshold, unjudged
            )
    for threshold in thresholds:
        scores[f"rr_t{threshold}"] = reciprocal_rank(
            ranking, grades, threshold, unjudged
        )
    return scores


def evaluate(
    rankings: Mapping[int, Sequence[ChunkRef]],
    grades: Mapping[int, Grades],
    ks: Sequence[int],
    unjudged: UnjudgedPolicy,
    thresholds: Sequence[int] = (1, 2),
) -> dict[int, Scores]:
    """Score every query's ranking. Queries with no judgements are skipped."""
    return {
        query_id: score_ranking(ranking, grades[query_id], ks, unjudged, thresholds)
        for query_id, ranking in rankings.items()
        if query_id in grades
    }


def defined(values: Iterable[float | None]) -> list[float]:
    """Drop the undefined values, which are never to be read as zero."""
    return [value for value in values if value is not None]


def macro_mean(
    per_query: Mapping[int, Scores], metric: str
) -> tuple[float | None, int]:
    """Return the mean of one metric across queries, and how many defined it.

    The count is returned rather than logged because a mean over 47 queries and
    a mean over 50 are different claims, and the difference has to travel with
    the number.
    """
    values = defined(scores.get(metric) for scores in per_query.values())
    return (statistics.fmean(values) if values else None, len(values))


def bootstrap_ci(
    per_query: Mapping[int, Scores],
    metric: str,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = 0.95,
) -> tuple[float, float] | None:
    """Percentile bootstrap interval for a metric's macro mean, over queries.

    Queries are the unit of resampling because they are what was sampled: 50 of
    them, many with only one or two relevant chunks, so a per-query score moves
    in large steps and a difference between two retrievers can easily be noise.
    An interval says how much of the estimate the sample actually supports.

    Returns None when fewer than two queries define the metric, where an
    interval would be arithmetic without meaning.
    """
    query_ids = [
        query_id
        for query_id, scores in per_query.items()
        if scores.get(metric) is not None
    ]
    if len(query_ids) < 2:
        return None

    values = {
        query_id: cast("float", per_query[query_id][metric]) for query_id in query_ids
    }
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        drawn = [values[rng.choice(query_ids)] for _ in query_ids]
        means.append(statistics.fmean(drawn))
    means.sort()
    tail = (1.0 - confidence) / 2.0
    low = means[int(tail * resamples)]
    high = means[min(int((1.0 - tail) * resamples), resamples - 1)]
    return (low, high)


@dataclass(frozen=True)
class PairedDifference:
    """One retriever's advantage over another on a metric, with its interval.

    `n` is the number of queries where both retrievers define the metric, which
    is the only set on which a difference exists. It can be smaller than either
    retriever's own n.
    """

    metric: str
    difference: float
    low: float
    high: float
    n: int

    @property
    def separable(self) -> bool:
        """Whether the interval excludes zero at the confidence it was built at.

        Not a p-value and not a claim of practical significance. It says the
        sample supports a difference in a direction, nothing more.
        """
        return self.low > 0.0 or self.high < 0.0


def paired_bootstrap(
    first: Mapping[int, Scores],
    second: Mapping[int, Scores],
    metric: str,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = 0.95,
) -> PairedDifference | None:
    """Bootstrap the difference between two retrievers on the same queries.

    Both retrievers are scored on every resample of the *same* query ids, so the
    query-to-query variation that they share cancels out of the difference. That
    is the whole point of pairing here: these four retrievers run over one
    corpus and one query set, and a query that is hard for one tends to be hard
    for all of them. Comparing two independent per-retriever intervals throws
    that away and will call real differences inseparable, because each interval
    is dominated by variation the comparison does not care about.

    Only queries where both retrievers define the metric are used; a difference
    against an undefined value has no meaning. Returns None when fewer than two
    such queries exist.
    """
    shared = sorted(
        query_id
        for query_id in set(first) & set(second)
        if first[query_id].get(metric) is not None
        and second[query_id].get(metric) is not None
    )
    if len(shared) < 2:
        return None

    deltas = {
        query_id: cast("float", first[query_id][metric])
        - cast("float", second[query_id][metric])
        for query_id in shared
    }
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        means.append(statistics.fmean([deltas[rng.choice(shared)] for _ in shared]))
    means.sort()
    tail = (1.0 - confidence) / 2.0
    return PairedDifference(
        metric=metric,
        difference=statistics.fmean(deltas.values()),
        low=means[int(tail * resamples)],
        high=means[min(int((1.0 - tail) * resamples), resamples - 1)],
        n=len(shared),
    )

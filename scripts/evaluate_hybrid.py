"""Score an RRF fusion of production bge-m3 and lexical search.

The protocol is `docs/HYBRID-PROTOCOL.md`, committed before this was run. Every
constant below is fixed there: k = 60 from Cormack, Clarke & Buettcher (2009),
depth 10 forced by the judged-pool depth, no weights, and NDCG@10 against
production as the single primary outcome.

Nothing here is tunable. There is no flag to change k, the depth or the weights,
because a flag is an invitation to try another value and report the better one.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import psycopg

from daedalus.embedding import embed_texts
from daedalus.evaluation.harness import (
    ChunkRef,
    Retriever,
    bootstrap_ci,
    evaluate,
    load_grades,
    load_queries,
    macro_mean,
    paired_bootstrap,
    run_retriever,
)
from daedalus.retrieval.search import lexical_search, vector_search
from daedalus.storage.database import connect

Connection = psycopg.Connection[tuple[object, ...]]

#: RRF constant. The published default, not a value fitted to this benchmark.
RRF_K = 60

#: Candidates taken from each retriever. Forced by the reference set: the pool
#: was judged to depth 10, so depth 11 would rank chunks nobody assessed.
DEPTH = 10

#: Cut-offs reported. Ten is the deepest the reference set can speak to.
KS = (1, 3, 5, 7, 10)

THRESHOLDS = (1, 2)

#: The single pre-declared primary outcome.
PRIMARY_METRIC = "ndcg@10"
PRIMARY_BASELINE = "bge-m3 (production)"


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ChunkRef]], k: int = RRF_K
) -> list[ChunkRef]:
    """Fuse ranked lists by reciprocal rank, best first.

    A chunk gets one term per list that returned it, summed. Appearing in two
    lists therefore beats appearing high in one: at k=60 and depth 10 the
    smallest two-list score, 2/70, exceeds the largest one-list score, 1/61. The
    protocol records that consequence rather than adjusting k to avoid it.

    Ties are broken on (doc_id, ordinal). They are reachable -- ranks (2,5) and
    (3,4) score identically -- and without a fixed key the fused order would not
    be reproducible.

    The result contains each chunk once, whatever its multiplicity across the
    inputs.
    """
    scores: dict[ChunkRef, float] = {}
    for ranking in rankings:
        for position, ref in enumerate(ranking, start=1):
            scores[ref] = scores.get(ref, 0.0) + 1.0 / (k + position)
    return sorted(scores, key=lambda ref: (-scores[ref], ref[0], ref[1]))


def vector_retriever(connection: Connection) -> Retriever:
    """Production bge-m3 over heading path plus chunk text."""

    def retrieve(text: str) -> list[ChunkRef]:
        vector = embed_texts([text], model="bge-m3")[0]
        return [
            (chunk.doc_id, chunk.ordinal)
            for chunk in vector_search(connection, vector, "bge-m3", DEPTH)
        ]

    return retrieve


def lexical_retriever(connection: Connection) -> Retriever:
    """PostgreSQL full-text search."""

    def retrieve(text: str) -> list[ChunkRef]:
        return [
            (chunk.doc_id, chunk.ordinal)
            for chunk in lexical_search(connection, text, DEPTH)
        ]

    return retrieve


def report(label: str, per_query: Mapping[int, dict[str, float | None]]) -> dict:
    """Print one variant's table and return it for the artifact."""
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    coverage, _ = macro_mean(per_query, f"unjudged@{DEPTH}")
    print(f"  unjudged in top {DEPTH}: {coverage:.2f} per query on average")
    print(f"\n  {'metric':12} {'mean':>7}  {'95% CI':>18}   n")

    names = [f"ndcg@{k}" for k in KS]
    for threshold in THRESHOLDS:
        names += [f"p@{k}_t{threshold}" for k in KS]
        names += [f"r@{k}_t{threshold}" for k in KS]
        names += [f"rr_t{threshold}"]

    rows: dict[str, object] = {}
    for name in names:
        mean, n = macro_mean(per_query, name)
        interval = bootstrap_ci(per_query, name)
        if mean is None:
            print(f"  {name:12} {'undefined':>7}   {'':18}   {n}")
            rows[name] = {"mean": None, "ci": None, "n": n}
            continue
        ci = f"[{interval[0]:.4f}, {interval[1]:.4f}]" if interval else ""
        print(f"  {name:12} {mean:7.4f}  {ci:>18}   {n}")
        rows[name] = {"mean": mean, "ci": list(interval) if interval else None, "n": n}
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results") / f"hybrid_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json",
    )
    parser.add_argument("--unjudged", choices=("zero", "skip"), required=True)
    args = parser.parse_args()

    with connect() as connection:
        queries = load_queries(connection)
        grades = load_grades(connection)
        print(
            f"queries {len(queries)}   judgements "
            f"{sum(len(g) for g in grades.values())}"
        )
        print(
            f"RRF k={RRF_K}   depth {DEPTH}   unweighted   "
            f"unjudged policy {args.unjudged}"
        )
        print(f"primary outcome: {PRIMARY_METRIC}, hybrid minus {PRIMARY_BASELINE}")

        vector = run_retriever(queries, vector_retriever(connection))
        lexical = run_retriever(queries, lexical_retriever(connection))
        hybrid = {
            query_id: reciprocal_rank_fusion([vector[query_id], lexical[query_id]])[
                :DEPTH
            ]
            for query_id in vector
        }

        scored = {
            label: evaluate(rankings, grades, KS, args.unjudged, THRESHOLDS)
            for label, rankings in (
                ("hybrid (RRF k=60)", hybrid),
                (PRIMARY_BASELINE, vector),
                ("lexical", lexical),
            )
        }

        artifact: dict[str, object] = {
            "generated_at": datetime.now(UTC).isoformat(),
            "protocol": "docs/HYBRID-PROTOCOL.md",
            "rrf_k": RRF_K,
            "depth": DEPTH,
            "weighted": False,
            "ks": list(KS),
            "thresholds": list(THRESHOLDS),
            "unjudged_policy": args.unjudged,
            "primary_metric": PRIMARY_METRIC,
            "primary_baseline": PRIMARY_BASELINE,
            "queries": len(queries),
            "judgements": sum(len(g) for g in grades.values()),
            "variants": {},
        }
        variants = artifact["variants"]
        assert isinstance(variants, dict)
        for label, per_query in scored.items():
            source = {
                "hybrid (RRF k=60)": hybrid,
                PRIMARY_BASELINE: vector,
                "lexical": lexical,
            }[label]
            variants[label] = {
                "aggregate": report(label, per_query),
                "per_query": {str(q): s for q, s in per_query.items()},
                "rankings": {
                    str(q): [list(ref) for ref in ranking]
                    for q, ranking in source.items()
                },
            }

        # --- the primary outcome, before anything secondary ------------------
        print(f"\n{'=' * 78}")
        print("PRIMARY OUTCOME")
        print("=" * 78)
        primary = paired_bootstrap(
            scored["hybrid (RRF k=60)"], scored[PRIMARY_BASELINE], PRIMARY_METRIC
        )
        assert primary is not None
        verdict = (
            "hybrid measurably better"
            if primary.separable and primary.difference > 0
            else "hybrid measurably worse"
            if primary.separable
            else "NULL RESULT - the interval includes zero"
        )
        print(f"  {PRIMARY_METRIC}, hybrid minus {PRIMARY_BASELINE}")
        print(
            f"    difference {primary.difference:+.4f}   "
            f"95% CI [{primary.low:+.4f}, {primary.high:+.4f}]   n={primary.n}"
        )
        print(f"    {verdict}")
        artifact["primary_outcome"] = {
            "metric": PRIMARY_METRIC,
            "difference": primary.difference,
            "ci": [primary.low, primary.high],
            "n": primary.n,
            "separable": primary.separable,
            "verdict": verdict,
        }

        # --- secondary, descriptive only -------------------------------------
        print(f"\n{'=' * 78}")
        print("SECONDARY (descriptive only, no decision weight)")
        print("=" * 78)
        secondary: dict[str, object] = {}
        labels = list(scored)
        for metric in ["ndcg@5", "ndcg@1", "rr_t1", "rr_t2", "r@10_t1", "r@10_t2"]:
            print(f"\n  {metric}")
            for i, first in enumerate(labels):
                for second in labels[i + 1 :]:
                    result = paired_bootstrap(scored[first], scored[second], metric)
                    if result is None:
                        continue
                    mark = " *" if result.separable else ""
                    print(
                        f"    {first + ' - ' + second:44} "
                        f"{result.difference:+8.4f} "
                        f"[{result.low:+8.4f}, {result.high:+8.4f}]  "
                        f"{result.n}{mark}"
                    )
                    secondary[f"{first} - {second} :: {metric}"] = {
                        "difference": result.difference,
                        "ci": [result.low, result.high],
                        "n": result.n,
                        "separable": result.separable,
                    }
        artifact["secondary"] = secondary
        print(
            "\n  * interval excludes zero. Six comparisons per metric: the "
            "family-wise\n    error rate is well above 5%, and none of these "
            "carries decision weight."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=1, sort_keys=True))
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

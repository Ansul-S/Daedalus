"""Score the four retrieval variants against the reference set.

Not a `daedalus` subcommand: this is evaluation infrastructure for one
experiment, not a feature of the product. Daedalus retrieves; this measures how
well, once, against a reference set specific to this corpus.

All four variants are scored by identical code against the same 50 queries and
the same judgements. Two of them built the judged pool and two did not, so the
unjudged count is reported beside every score. When it is zero the pool covers
the ranking completely and the unjudged policy cannot affect the result; when it
is not, the numbers are over the judged subset and have to be read that way.

Nothing here tunes anything. There is no parameter to fit, no threshold chosen
after the fact, and no variant selected on the strength of its score.
"""

from __future__ import annotations

import argparse
import json
import sys
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

#: Cut-offs to report. Ten is the depth every retriever's pool was judged to;
#: beyond it the reference set cannot say whether a result is relevant.
KS = (1, 3, 5, 7, 10)

#: Both binary cuts are reported. Grade 1 is useful evidence, grade 2 is
#: evidence sufficient to answer, and the difference is the point of the scale.
THRESHOLDS = (1, 2)

#: Deepest ranking requested. Equal to max(KS) by design, not by coincidence.
DEPTH = max(KS)

#: Metrics whose pairwise differences are printed. Every metric is written to
#: the JSON, but printing all of them across six pairs would be a wall of
#: intervals inviting exactly the cherry-picking this experiment must not do.
HEADLINE = ("ndcg@10", "ndcg@5", "rr_t1", "r@10_t1", "r@10_t2")

#: (label, storage key for the embeddings, model to call for the query vector)
VECTOR_VARIANTS = (
    ("bge-m3 (production)", "bge-m3", "bge-m3"),
    ("bge-m3-noheading", "bge-m3-noheading", "bge-m3"),
    ("all-minilm", "all-minilm", "all-minilm"),
)


def vector_retriever(connection: Connection, key: str, model: str) -> Retriever:
    """A retriever over one stored embedding set.

    The key and the model are separate because they differ for the heading
    ablation: it calls bge-m3 but reads vectors stored under another name. A
    query embedded with one model and searched against another's vectors raises
    a dimension error rather than returning a plausible wrong ranking, which is
    the failure mode worth having.
    """

    def retrieve(text: str) -> list[ChunkRef]:
        vector = embed_texts([text], model=model)[0]
        return [
            (chunk.doc_id, chunk.ordinal)
            for chunk in vector_search(connection, vector, key, DEPTH)
        ]

    return retrieve


def lexical_retriever(connection: Connection) -> Retriever:
    """A retriever over PostgreSQL full-text search."""

    def retrieve(text: str) -> list[ChunkRef]:
        return [
            (chunk.doc_id, chunk.ordinal)
            for chunk in lexical_search(connection, text, DEPTH)
        ]

    return retrieve


def report(
    label: str, per_query: dict[int, dict[str, float | None]]
) -> dict[str, object]:
    """Print one variant's table and return it for the JSON artifact."""
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    coverage = macro_mean(per_query, f"unjudged@{DEPTH}")[0]
    print(f"  unjudged in top {DEPTH}: {coverage:.2f} per query on average")
    print(f"\n  {'metric':12} {'mean':>7}  {'95% CI':>18}   n")

    rows: dict[str, object] = {}
    names = [f"ndcg@{k}" for k in KS]
    for threshold in THRESHOLDS:
        names += [f"p@{k}_t{threshold}" for k in KS]
        names += [f"r@{k}_t{threshold}" for k in KS]
        names += [f"rr_t{threshold}"]
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


def pairwise(
    scored: dict[str, dict[int, dict[str, float | None]]],
) -> dict[str, object]:
    """Paired differences between every pair of retrievers.

    Paired rather than a comparison of two independent intervals: all four run
    over the same queries, a query hard for one tends to be hard for all, and
    pairing removes that shared variation from the difference.
    """
    labels = list(scored)
    out: dict[str, object] = {}

    print(f"\n{'=' * 78}")
    print("PAIRWISE DIFFERENCES (paired bootstrap, 95%, 10,000 resamples)")
    print("=" * 78)
    print("  An interval excluding zero means the sample supports a difference")
    print("  in that direction. It is not a p-value, and with 6 pairs per metric")
    print("  the family-wise error rate is well above 5% -- read a single")
    print("  marginal interval accordingly.")

    for metric in HEADLINE:
        print(f"\n  {metric}")
        print(f"    {'A - B':44} {'diff':>8} {'95% CI':>20}  n")
        for i, first in enumerate(labels):
            for second in labels[i + 1 :]:
                result = paired_bootstrap(scored[first], scored[second], metric)
                key = f"{first} - {second} :: {metric}"
                if result is None:
                    print(f"    {first + ' - ' + second:44} {'undefined':>8}")
                    out[key] = None
                    continue
                mark = " *" if result.separable else ""
                print(
                    f"    {first + ' - ' + second:44} {result.difference:8.4f} "
                    f"[{result.low:8.4f}, {result.high:8.4f}]  {result.n}{mark}"
                )
                out[key] = {
                    "difference": result.difference,
                    "ci": [result.low, result.high],
                    "n": result.n,
                    "separable": result.separable,
                }

    # Everything, not just the headline, goes to the artifact.
    for metric in sorted({m for s in scored.values() for q in s.values() for m in q}):
        for i, first in enumerate(labels):
            for second in labels[i + 1 :]:
                key = f"{first} - {second} :: {metric}"
                if key in out:
                    continue
                result = paired_bootstrap(scored[first], scored[second], metric)
                out[key] = (
                    None
                    if result is None
                    else {
                        "difference": result.difference,
                        "ci": [result.low, result.high],
                        "n": result.n,
                        "separable": result.separable,
                    }
                )
    print("\n  * interval excludes zero")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results") / f"retrieval_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json",
        help="where to write the JSON artifact",
    )
    parser.add_argument("--unjudged", choices=("zero", "skip"), required=True)
    args = parser.parse_args()

    with connect() as connection:
        queries = load_queries(connection)
        grades = load_grades(connection)
        print(
            f"queries {len(queries)}   judged queries {len(grades)}   "
            f"judgements {sum(len(g) for g in grades.values())}"
        )
        print(f"unjudged policy: {args.unjudged}   K {KS}   thresholds {THRESHOLDS}")

        retrievers: list[tuple[str, Retriever]] = [
            (label, vector_retriever(connection, key, model))
            for label, key, model in VECTOR_VARIANTS
        ]
        retrievers.append(("lexical", lexical_retriever(connection)))

        artifact: dict[str, object] = {
            "generated_at": datetime.now(UTC).isoformat(),
            "queries": len(queries),
            "judgements": sum(len(g) for g in grades.values()),
            "ks": list(KS),
            "thresholds": list(THRESHOLDS),
            "unjudged_policy": args.unjudged,
            "depth": DEPTH,
            "variants": {},
        }
        scored: dict[str, dict[int, dict[str, float | None]]] = {}
        for label, retriever in retrievers:
            rankings = run_retriever(queries, retriever)
            per_query = evaluate(rankings, grades, KS, args.unjudged, THRESHOLDS)
            scored[label] = per_query
            variants = artifact["variants"]
            assert isinstance(variants, dict)
            variants[label] = {
                "aggregate": report(label, per_query),
                "per_query": {
                    str(query_id): scores for query_id, scores in per_query.items()
                },
                "rankings": {
                    str(query_id): [list(ref) for ref in ranking]
                    for query_id, ranking in rankings.items()
                },
            }

        artifact["pairwise"] = pairwise(scored)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=1, sort_keys=True))
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

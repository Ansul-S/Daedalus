"""
Run the retrieval benchmark.

    uv run python -m daedalus.evaluation.run --embedder bge
    uv run python -m daedalus.evaluation.run --embedder fake --split dev

Reads only the frozen text and the labelled dataset, so a run needs no
source documents, no OCR, and no vision model. The database is in-memory
and thrown away: a benchmark that mutates the developer's index would
change what it is measuring.
"""

from __future__ import annotations

import argparse

from daedalus.config import constants
from daedalus.core.logging import configure_logging
from daedalus.db import connect, initialize_schema
from daedalus.embeddings import BGEEmbedder, FakeEmbedder
from daedalus.evaluation.corpus import load_frozen
from daedalus.evaluation.dataset import load_dataset, verify_anchors
from daedalus.evaluation.harness import RunResult, evaluate, index_frozen
from daedalus.interfaces.embedding import Embedder
from daedalus.retrieval import DenseRetriever, HybridRetriever, LexicalRetriever


def _format(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:.3f}"


def _print_table(results: list[RunResult]) -> None:
    print(f"\n{'retriever':<10} {'recall@k':>9} {'mrr':>7} {'ndcg@k':>8}")
    print("-" * 37)

    for result in results:
        summary = result.summary()
        print(
            f"{result.retriever:<10} {_format(summary['recall']):>9} "
            f"{_format(summary['mrr']):>7} {_format(summary['ndcg']):>8}"
        )


def _print_slices(results: list[RunResult], attribute: str) -> None:
    names = sorted({name for result in results for name in result.sliced_by(attribute)})

    print(f"\nrecall@k by {attribute}")
    header = f"{'':<14}" + "".join(f"{result.retriever:>10}" for result in results)
    print(header)
    print("-" * len(header))

    for name in names:
        row = f"{name:<14}"
        for result in results:
            row += f"{_format(result.sliced_by(attribute).get(name, {}).get('recall')):>10}"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the retrieval benchmark")
    parser.add_argument("--embedder", choices=["fake", "bge"], default="fake")
    parser.add_argument("--k", type=int, default=constants.DEFAULT_TOP_K)
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--chunk-size", type=int, default=constants.DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=constants.DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--candidates", type=int, default=constants.HYBRID_CANDIDATES)
    arguments = parser.parse_args()

    configure_logging()

    documents = load_frozen()
    queries = load_dataset()

    # Refuse to report numbers from a drifted dataset. A mis-anchored label
    # still produces a plausible score, which is the dangerous kind of wrong.
    problems = verify_anchors(queries, documents)

    if problems:
        raise SystemExit("dataset does not match the frozen corpus:\n  " + "\n  ".join(problems))

    if arguments.split != "all":
        queries = [query for query in queries if query.split == arguments.split]

    embedder: Embedder = FakeEmbedder() if arguments.embedder == "fake" else BGEEmbedder()

    connection = connect(":memory:")
    initialize_schema(connection)

    corpus = index_frozen(
        connection,
        documents,
        embedder,
        chunk_size=arguments.chunk_size,
        overlap=arguments.overlap,
    )

    unverified = sum(1 for query in queries if query.verified_by != "human")

    print(
        f"\n{len(documents)} documents, {corpus.n_chunks} chunks "
        f"(size {arguments.chunk_size}/{arguments.overlap})"
    )
    print(
        f"{len(queries)} queries on split {arguments.split!r}, "
        f"embedder {arguments.embedder!r}, k={arguments.k}, candidates={arguments.candidates}"
    )

    if unverified:
        print(f"\n  WARNING: {unverified}/{len(queries)} labels are not human-verified.")
        print("  Numbers below are a smoke test of the harness, not a measurement.")

    retrievers = [
        ("lexical", LexicalRetriever(connection)),
        ("dense", DenseRetriever(connection, embedder)),
        (
            "hybrid",
            HybridRetriever(connection, embedder, candidates=arguments.candidates),
        ),
    ]

    results = [
        evaluate(retriever, corpus, queries, name=name, k=arguments.k)
        for name, retriever in retrievers
    ]

    _print_table(results)
    _print_slices(results, "query_type")
    _print_slices(results, "source_type")

    unanswerable = results[0].unanswerable_returning_results
    print(
        f"\nunanswerable queries still returning chunks: "
        f"{unanswerable}/{sum(1 for q in queries if not q.answerable)}"
        " — retrieval always has a nearest neighbour; refusing is the generator's job\n"
    )

    connection.close()


if __name__ == "__main__":
    main()

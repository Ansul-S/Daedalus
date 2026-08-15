"""
Running the benchmark.

Indexes the frozen text — never the source documents — so a run needs no
OCR, no vision model, and no GPU, and produces the same numbers on a
laptop and in CI. Chunking happens here rather than being frozen too,
because chunk size is one of the things being measured.

The ablation this exists to serve is a loop over retrievers. All three
satisfy one port, so swapping them changes nothing else, and a difference
in the numbers is attributable to the retrieval strategy rather than to
the code around it.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field

from daedalus.config import constants
from daedalus.evaluation.corpus import FrozenDocument
from daedalus.evaluation.dataset import EvalQuery
from daedalus.evaluation.metrics import mean, mrr, ndcg_at_k, recall_at_k
from daedalus.evaluation.spans import DEFAULT_MIN_OVERLAP, resolve_spans
from daedalus.ingestion.chunker import chunk_text
from daedalus.ingestion.types import Chunk
from daedalus.interfaces.embedding import Embedder
from daedalus.interfaces.retrieval import Retriever
from daedalus.storage import chunks as chunk_store
from daedalus.storage import documents as document_store

__all__ = ["IndexedCorpus", "QueryScore", "RunResult", "evaluate", "index_frozen"]


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexedCorpus:
    """The frozen corpus, chunked and indexed, with a map back to labels."""

    chunks_by_doc: dict[str, list[Chunk]]
    ids_by_doc: dict[str, list[int]]

    @property
    def n_chunks(self) -> int:
        return sum(len(chunks) for chunks in self.chunks_by_doc.values())

    def relevant_for(
        self, query: EvalQuery, *, min_overlap: float = DEFAULT_MIN_OVERLAP
    ) -> dict[int, int]:
        """Resolve a query's labelled spans to ``{chunk_id: grade}``."""

        grades: dict[int, int] = {}

        for doc_id in {span.doc_id for span in query.relevant_spans}:
            if doc_id not in self.chunks_by_doc:
                logger.warning("%s references unindexed document %s", query.id, doc_id)
                continue

            resolved = resolve_spans(
                query.spans(),
                self.chunks_by_doc[doc_id],
                self.ids_by_doc[doc_id],
                doc_id=doc_id,
                min_overlap=min_overlap,
            )

            for chunk_id, grade in resolved.items():
                grades[chunk_id] = max(grades.get(chunk_id, 0), grade)

        return grades


def index_frozen(
    connection: sqlite3.Connection,
    documents: Sequence[FrozenDocument],
    embedder: Embedder,
    *,
    chunk_size: int = constants.DEFAULT_CHUNK_SIZE,
    overlap: int = constants.DEFAULT_CHUNK_OVERLAP,
) -> IndexedCorpus:
    """Chunk, embed, and index the frozen corpus into an open database."""

    chunks_by_doc: dict[str, list[Chunk]] = {}
    ids_by_doc: dict[str, list[int]] = {}

    for document in documents:
        document_store.create(
            connection,
            doc_id=document.doc_id,
            filename=document.filename,
            source_type=document.source_type,
            content_hash=document.text_sha256,
        )

        chunks = chunk_text(
            document.text, document.segments, chunk_size=chunk_size, overlap=overlap
        )
        embeddings = embedder.embed_documents([chunk.text for chunk in chunks])

        chunk_store.replace(connection, document.doc_id, chunks, embeddings)

        rows = connection.execute(
            "SELECT id FROM chunks WHERE doc_id = ? ORDER BY ordinal", (document.doc_id,)
        ).fetchall()

        chunks_by_doc[document.doc_id] = chunks
        ids_by_doc[document.doc_id] = [row["id"] for row in rows]

        logger.info("Indexed %s: %d chunks", document.doc_id, len(chunks))

    return IndexedCorpus(chunks_by_doc=chunks_by_doc, ids_by_doc=ids_by_doc)


@dataclass(frozen=True)
class QueryScore:
    """What one query scored under one retriever."""

    query_id: str
    query_type: str
    source_type: str
    answerable: bool
    n_relevant: int
    n_retrieved: int
    recall: float | None
    reciprocal_rank: float | None
    ndcg: float | None


@dataclass
class RunResult:
    """Every query's score for one retriever."""

    retriever: str
    k: int
    scores: list[QueryScore] = field(default_factory=list)

    def summary(self, subset: Sequence[QueryScore] | None = None) -> dict[str, float | None]:
        """Averaged metrics over all scores, or over a slice of them."""

        rows = self.scores if subset is None else subset

        return {
            "recall": mean([row.recall for row in rows]),
            "mrr": mean([row.reciprocal_rank for row in rows]),
            "ndcg": mean([row.ndcg for row in rows]),
        }

    def sliced_by(self, attribute: str) -> dict[str, dict[str, float | None]]:
        """Metrics grouped by an attribute such as ``query_type``."""

        groups: dict[str, list[QueryScore]] = {}

        for score in self.scores:
            groups.setdefault(getattr(score, attribute), []).append(score)

        return {name: self.summary(rows) for name, rows in sorted(groups.items())}

    @property
    def unanswerable_returning_results(self) -> int:
        """
        Unanswerable queries that still got chunks back.

        Not a retrieval failure — dense search always has a nearest
        neighbour — but it is the population the generator has to refuse
        over, so it is worth having the count in front of you.
        """

        return sum(1 for score in self.scores if not score.answerable and score.n_retrieved)


def evaluate(
    retriever: Retriever,
    corpus: IndexedCorpus,
    queries: Sequence[EvalQuery],
    *,
    name: str,
    k: int = constants.DEFAULT_TOP_K,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
) -> RunResult:
    """Run every query through one retriever and score the results."""

    result = RunResult(retriever=name, k=k)

    for query in queries:
        relevant = corpus.relevant_for(query, min_overlap=min_overlap)
        retrieved = [hit.chunk_id for hit in retriever.search(query.query, k)]

        result.scores.append(
            QueryScore(
                query_id=query.id,
                query_type=query.query_type,
                source_type=query.source_type,
                answerable=query.answerable,
                n_relevant=len(relevant),
                n_retrieved=len(retrieved),
                recall=recall_at_k(retrieved, relevant, k),
                reciprocal_rank=mrr(retrieved, relevant),
                ndcg=ndcg_at_k(retrieved, relevant, k),
            )
        )

    return result

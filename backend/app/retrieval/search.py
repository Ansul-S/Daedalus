"""Hybrid search: pgvector similarity and Postgres full-text search, merged with RRF.

Vectors find paraphrases ("why scale the attention scores"); keywords find exact terms
("AdamW", "ntotal") that embeddings blur.

Both retrievers read the current chunks only. The chunks a re-ingestion replaced stay in the
table for the questions that cite them, but they are no longer material to search.
"""

from dataclasses import dataclass
from typing import Literal

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document
from app.llm.embeddings import Embedder, EmbeddingError
from app.retrieval.fusion import reciprocal_rank_fusion

SearchMode = Literal["hybrid", "vector", "keyword"]

# Candidates taken from each retriever
CANDIDATES = 50
# In hybrid search, only the full-text ranking's top ten join the fusion, at half the vector
# ranking's weight. The OR query matches nearly every passage, so at full share a passage in
# the middle of both lists outranked one high in the vector list alone: a question whose answer
# was 7th by vector and missing from full-text came 41st. Chosen on 27 questions and judged
# once on 45 others (make eval-retrieval): MRR@5 on the asked questions 0.870 -> 0.893.
KEYWORD_CANDIDATES = 10
KEYWORD_WEIGHT = 0.5

# Questions are long, and requiring every word (AND) would match almost nothing, so the
# stemmed words are ORed. ts_rank_cd adds up the matches, with section words weighing more
# than body text, and normalization 2 divides the sum by the chunk's length: otherwise a
# long code chunk that repeats a few of the words outranks a short passage about them.
_KEYWORD_QUERY = text(
    """
    WITH query AS (
        SELECT replace(plainto_tsquery('english', :query)::text, ' & ', ' | ')::tsquery AS terms
    )
    SELECT chunks.id
    FROM chunks, query
    WHERE chunks.search_vector @@ query.terms AND chunks.superseded_at IS NULL
    ORDER BY ts_rank_cd(chunks.search_vector, query.terms, 2) DESC, chunks.id
    LIMIT :limit
    """
)


@dataclass
class Hit:
    chunk: Chunk
    document: Document
    score: float
    vector_rank: int | None
    keyword_rank: int | None


@dataclass
class SearchResult:
    mode: SearchMode
    hits: list[Hit]
    warning: str | None = None


def fuse_rankings(
    vector_ids: list[int],
    keyword_ids: list[int],
    keyword_candidates: int = KEYWORD_CANDIDATES,
    keyword_weight: float = KEYWORD_WEIGHT,
) -> list[tuple[int, float]]:
    """Hybrid ranking. With one of the two rankings empty, the other is kept whole."""
    if not vector_ids:
        return reciprocal_rank_fusion([keyword_ids] if keyword_ids else [])
    if not keyword_ids:
        return reciprocal_rank_fusion([vector_ids])
    return reciprocal_rank_fusion(
        [vector_ids, keyword_ids[:keyword_candidates]], weights=[1.0, keyword_weight]
    )


async def vector_ranking(session: AsyncSession, vector: list[float], limit: int) -> list[int]:
    # The HNSW index returns at most ef_search rows (default 40).
    await session.execute(select(func.set_config("hnsw.ef_search", str(max(limit, 40)), True)))
    ids = await session.scalars(
        select(Chunk.id)
        .where(Chunk.superseded_at.is_(None))
        .order_by(Chunk.embedding.cosine_distance(vector))
        .limit(limit)
    )
    return list(ids)


async def keyword_ranking(session: AsyncSession, query: str, limit: int) -> list[int]:
    ids = await session.scalars(_KEYWORD_QUERY, {"query": query, "limit": limit})
    return list(ids)


async def search(
    session: AsyncSession,
    query: str,
    *,
    limit: int = 10,
    mode: SearchMode = "hybrid",
    embedder: Embedder | None = None,
) -> SearchResult:
    warning = None
    vector_ids: list[int] = []
    keyword_ids: list[int] = []

    if mode != "keyword":
        try:
            if embedder is None:
                raise EmbeddingError("semantic search needs the local embedding model")
            vector_ids = await vector_ranking(
                session, await embedder.embed_query(query), CANDIDATES
            )
        except EmbeddingError as exc:
            if mode == "vector":
                raise
            warning = f"{exc}; showing keyword matches only"
            mode = "keyword"
    if mode != "vector":
        keyword_ids = await keyword_ranking(session, query, CANDIDATES)

    fused = fuse_rankings(vector_ids, keyword_ids)[:limit]
    rows = await session.execute(
        select(Chunk, Document)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.id.in_([chunk_id for chunk_id, _ in fused]))
    )
    found = {chunk.id: (chunk, document) for chunk, document in rows.tuples()}
    vector_ranks = {chunk_id: rank for rank, chunk_id in enumerate(vector_ids, start=1)}
    keyword_ranks = {chunk_id: rank for rank, chunk_id in enumerate(keyword_ids, start=1)}
    hits = [
        Hit(
            chunk=found[chunk_id][0],
            document=found[chunk_id][1],
            score=score,
            vector_rank=vector_ranks.get(chunk_id),
            keyword_rank=keyword_ranks.get(chunk_id),
        )
        for chunk_id, score in fused
        if chunk_id in found
    ]
    return SearchResult(mode=mode, hits=hits, warning=warning)

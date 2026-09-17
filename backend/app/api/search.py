from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document
from app.db.session import get_session
from app.ingest.chunking import SECTION_SEPARATOR
from app.llm.embeddings import Embedder, EmbeddingError
from app.retrieval.search import SearchMode, search

router = APIRouter(tags=["search"])


class SearchHitOut(BaseModel):
    chunk_id: int
    document_id: int
    document_title: str
    source_type: str
    citation: str
    link: str | None
    section: str | None
    page_start: int | None
    page_end: int | None
    cell_start: int | None
    cell_end: int | None
    content_types: list[str]
    text: str
    score: float
    vector_rank: int | None
    keyword_rank: int | None


class SearchOut(BaseModel):
    query: str
    # The mode actually used: hybrid falls back to keyword when the embedding model is down
    mode: SearchMode
    warning: str | None
    results: list[SearchHitOut]


def _span(singular: str, plural: str, start: int, end: int | None) -> str:
    return f"{singular} {start}" if end in (None, start) else f"{plural} {start}–{end}"


def citation(document: Document, chunk: Chunk) -> str:
    """E.g. "RNN Intuition, pp. 7–8", "Notebook, cells 188–193", "Paper, § 3.2.1 Attention"."""
    if chunk.page_start is not None:
        where = _span("p.", "pp.", chunk.page_start, chunk.page_end)
    elif chunk.cell_start is not None:
        where = _span("cell", "cells", chunk.cell_start, chunk.cell_end)
    elif chunk.section:
        where = f"§ {chunk.section.split(SECTION_SEPARATOR)[-1]}"
    else:
        return document.title
    return f"{document.title}, {where}"


def source_link(document: Document, chunk: Chunk) -> str | None:
    """A link to the exact place in an online source (arXiv section or PDF page)."""
    if document.url is None:
        return None
    if chunk.anchor:
        return f"{document.url}#{chunk.anchor}"
    if chunk.page_start is not None:
        return f"{document.url}#page={chunk.page_start}"
    return document.url


def get_embedder(request: Request) -> Embedder | None:
    return getattr(request.app.state, "embedder", None)


@router.get("/search")
async def search_chunks(
    q: Annotated[str, Query(min_length=1, max_length=2000, description="Question or keywords")],
    session: Annotated[AsyncSession, Depends(get_session)],
    embedder: Annotated[Embedder | None, Depends(get_embedder)],
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    mode: SearchMode = "hybrid",
) -> SearchOut:
    """Search all ingested material. `mode=vector` or `mode=keyword` runs a single retriever."""
    try:
        result = await search(session, q, limit=limit, mode=mode, embedder=embedder)
    except EmbeddingError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    return SearchOut(
        query=q,
        mode=result.mode,
        warning=result.warning,
        results=[
            SearchHitOut(
                chunk_id=hit.chunk.id,
                document_id=hit.document.id,
                document_title=hit.document.title,
                source_type=hit.document.source_type,
                citation=citation(hit.document, hit.chunk),
                link=source_link(hit.document, hit.chunk),
                section=hit.chunk.section,
                page_start=hit.chunk.page_start,
                page_end=hit.chunk.page_end,
                cell_start=hit.chunk.cell_start,
                cell_end=hit.chunk.cell_end,
                content_types=hit.chunk.content_types,
                text=hit.chunk.text,
                score=round(hit.score, 5),
                vector_rank=hit.vector_rank,
                keyword_rank=hit.keyword_rank,
            )
            for hit in result.hits
        ],
    )

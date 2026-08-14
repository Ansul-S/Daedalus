"""
Search over indexed material.

The retrieval half of the vertical slice, exposed. Hits come back as
ranked chunk ids, which this module hydrates into text plus the source
offsets a citation needs.

``POST`` rather than ``GET`` because a query is a body, not an identifier:
study questions are long, contain characters that would need escaping in a
path, and should not end up in proxy access logs.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends

from daedalus.api.dependencies import get_db, get_embedder
from daedalus.api.schemas import SearchRequest, SearchResponse, SearchResult
from daedalus.interfaces.embedding import Embedder
from daedalus.retrieval import HybridRetriever
from daedalus.storage import chunks as chunk_store

__all__ = ["router"]


logger = logging.getLogger(__name__)


router = APIRouter(tags=["search"])


@router.post("/search")
def search(
    request: SearchRequest,
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> SearchResponse:
    """
    Return the chunks most relevant to a query.

    An empty result is a normal answer, not an error — the corpus
    genuinely may not cover the question, and reporting that honestly is
    what the unanswerable slice of the evaluation set measures.
    """

    hits = HybridRetriever(db, embedder).search(request.query, request.top_k)

    # One round trip for the whole ranking, then reordered to match it.
    records = chunk_store.fetch(db, [hit.chunk_id for hit in hits])
    scores = {hit.chunk_id: hit.score for hit in hits}

    logger.info("Query %r returned %d results", request.query, len(records))

    return SearchResponse(
        query=request.query,
        results=[SearchResult.from_record(record, scores[record.id]) for record in records],
    )

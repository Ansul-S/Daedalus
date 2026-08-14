"""
Request and response models.

These are the API's contract with a client, and they are deliberately not
the same types as ``storage.types``. Those mirror database columns; these
mirror what a caller is allowed to send and promised to receive. Merging
them would mean a schema change to the database silently becoming a
breaking change to the API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from daedalus.config import constants
from daedalus.generation import Answer, Source
from daedalus.storage.types import ChunkRecord, DocumentRecord

__all__ = [
    "AnswerRequest",
    "AnswerResponse",
    "DocumentResponse",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "SourceResponse",
]


class DocumentResponse(BaseModel):
    """A document's ingestion state, as reported by the status endpoint."""

    id: str
    filename: str
    source_type: str
    status: str
    error: str | None = None
    n_pages: int | None = None
    n_chunks: int | None = None
    created_at: str
    updated_at: str

    @classmethod
    def from_record(cls, record: DocumentRecord) -> DocumentResponse:
        # content_hash is deliberately not exposed: it is an internal
        # deduplication key, and nothing a client can do with it is useful.
        return cls(
            id=record.id,
            filename=record.filename,
            source_type=record.source_type,
            status=record.status,
            error=record.error,
            n_pages=record.n_pages,
            n_chunks=record.n_chunks,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )


class SearchRequest(BaseModel):
    """A query against the index."""

    query: str = Field(min_length=1, max_length=1000)

    # Bounded because top_k drives a brute-force KNN scan and the size of
    # the context a generated answer would later be grounded in.
    top_k: int = Field(default=constants.DEFAULT_TOP_K, ge=1, le=50)


class SearchResult(BaseModel):
    """
    One retrieved chunk.

    Carries the source offsets as well as the text: they are what a
    citation resolves to, and what the evaluation harness scores against.
    """

    chunk_id: int
    doc_id: str
    ordinal: int
    text: str
    score: float
    extraction: str
    source_start: int
    source_end: int
    page: int | None = None

    @classmethod
    def from_record(cls, record: ChunkRecord, score: float) -> SearchResult:
        return cls(
            chunk_id=record.id,
            doc_id=record.doc_id,
            ordinal=record.ordinal,
            text=record.text,
            score=score,
            extraction=record.extraction,
            source_start=record.source_start,
            source_end=record.source_end,
            page=record.page,
        )


class SearchResponse(BaseModel):
    """Results for one query, best first."""

    query: str
    results: list[SearchResult]


class AnswerRequest(BaseModel):
    """A question to answer from the indexed material."""

    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=constants.DEFAULT_TOP_K, ge=1, le=20)


class SourceResponse(BaseModel):
    """A numbered source, as offered to the model."""

    index: int
    chunk_id: int
    doc_id: str
    ordinal: int
    text: str
    extraction: str
    source_start: int
    source_end: int
    page: int | None = None

    @classmethod
    def from_source(cls, source: Source) -> SourceResponse:
        return cls(
            index=source.index,
            chunk_id=source.chunk_id,
            doc_id=source.doc_id,
            ordinal=source.ordinal,
            text=source.text,
            extraction=source.extraction,
            source_start=source.source_start,
            source_end=source.source_end,
            page=source.page,
        )


class AnswerResponse(BaseModel):
    """
    A grounded answer.

    Both lists are returned: ``citations`` is what the model referenced,
    ``sources`` is everything it was shown. A client renders the first and
    a reviewer inspects the gap between them.
    """

    question: str
    answer: str
    model: str
    citations: list[SourceResponse]
    sources: list[SourceResponse]

    @classmethod
    def from_answer(cls, answer: Answer) -> AnswerResponse:
        return cls(
            question=answer.question,
            answer=answer.text,
            model=answer.model,
            citations=[SourceResponse.from_source(source) for source in answer.citations],
            sources=[SourceResponse.from_source(source) for source in answer.sources],
        )

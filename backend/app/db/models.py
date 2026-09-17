"""Database tables: source documents, their searchable chunks, and ingestion jobs."""

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIMENSIONS = 1024

SOURCE_TYPES = ("pdf", "notebook", "arxiv")
DOCUMENT_STATUSES = ("pending", "ready", "failed")
JOB_STATUSES = ("queued", "running", "done", "failed")
CONTENT_TYPES = ("text", "code", "formula", "table")


def _one_of(column: str, values: tuple[str, ...]) -> CheckConstraint:
    allowed = ", ".join(f"'{value}'" for value in values)
    return CheckConstraint(f"{column} IN ({allowed})", name=f"{column}_valid")


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_type: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    authors: Mapped[str | None] = mapped_column(Text)
    # Original file name of an upload
    filename: Mapped[str | None] = mapped_column(Text)
    # Stored file, relative to the data directory
    path: Mapped[str | None] = mapped_column(Text)
    sha256: Mapped[str | None] = mapped_column(Text, unique=True)
    arxiv_id: Mapped[str | None] = mapped_column(Text, unique=True)
    arxiv_version: Mapped[str | None] = mapped_column(Text)
    # License URL as reported by arXiv; decides whether a paper may be shown publicly
    license: Mapped[str | None] = mapped_column(Text)
    # Where the ingested text came from (e.g. the arXiv HTML page)
    url: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, server_default="pending")
    error: Mapped[str | None] = mapped_column(Text)
    # Abstract, categories, page count, parser details and timings
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", passive_deletes=True, order_by="Chunk.position"
    )
    jobs: Mapped[list["Job"]] = relationship(back_populates="document", passive_deletes=True)

    __table_args__ = (
        _one_of("source_type", SOURCE_TYPES),
        _one_of("status", DOCUMENT_STATUSES),
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    # Order within the document, starting at 0
    position: Mapped[int]
    # Heading path, e.g. "3 Model Architecture > 3.2 Attention"
    section: Mapped[str | None] = mapped_column(Text)
    # Which of CONTENT_TYPES the chunk contains
    content_types: Mapped[list[str]] = mapped_column(ARRAY(Text))
    # Markdown: code in fenced blocks, math as $...$ / $$...$$ LaTeX
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int]
    # Citation locators, all 1-based: PDF pages or notebook cells
    page_start: Mapped[int | None]
    page_end: Mapped[int | None]
    cell_start: Mapped[int | None]
    cell_end: Mapped[int | None]
    # Fragment identifier of the section in an HTML source (arXiv), e.g. "S3.SS2"
    anchor: Mapped[str | None] = mapped_column(Text)
    # Half precision: half the storage of `vector`, and float16 is plenty for cosine ranking
    embedding: Mapped[list[float]] = mapped_column(HALFVEC(EMBEDDING_DIMENSIONS))
    # Full-text index input: the heading path ranks above the body text
    search_vector: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed(
            "setweight(to_tsvector('english', coalesce(section, '')), 'A') || "
            "setweight(to_tsvector('english', text), 'B')",
            persisted=True,
        ),
    )

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("document_id", "position"),
        Index(
            "chunks_embedding_idx",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "halfvec_cosine_ops"},
        ),
        Index("chunks_search_vector_idx", "search_vector", postgresql_using="gin"),
    )


class Job(Base):
    """One ingestion run for a document. Postgres doubles as the job queue."""

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[str] = mapped_column(Text, server_default="queued")
    # Parser options, e.g. {"ocr": false, "formulas": true}
    options: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    # Current step, e.g. "embedding 40/120"
    progress: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    document: Mapped[Document] = relationship(back_populates="jobs")

    __table_args__ = (
        _one_of("status", JOB_STATUSES),
        # Keeps "find the oldest queued job" cheap however many finished jobs pile up
        Index("jobs_queued_idx", "created_at", postgresql_where=text("status = 'queued'")),
    )

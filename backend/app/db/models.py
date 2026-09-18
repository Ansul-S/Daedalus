"""Database tables: source documents, their searchable chunks and ingestion jobs, the topic
map built over the chunks, and the generated questions."""

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    ColumnElement,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Text,
    UniqueConstraint,
    func,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIMENSIONS = 1024

SOURCE_TYPES = ("pdf", "notebook", "arxiv")
DOCUMENT_STATUSES = ("pending", "ready", "failed")
JOB_STATUSES = ("queued", "running", "done", "failed")
JOB_KINDS = ("ingest", "generate")
TASK_STATUSES = ("queued", "running", "done", "failed")
CONTENT_TYPES = ("text", "code", "formula", "table")
QUESTION_STATUSES = ("accepted", "rejected", "retired")
# The shapes a question can take, from the design notes: an intuition check, a why or how
# explanation, a comparison, a trade-off, a failure mode, a link between two concepts that
# sit in different chunks, or a question about a paper's problem, idea, limits and extensions
QUESTION_STYLES = (
    "intuition",
    "why_how",
    "compare",
    "tradeoffs",
    "failure_modes",
    "connection",
    "paper",
)

# Chunks left behind by an earlier ingestion are excluded from search and generation
CURRENT_CHUNKS = text("superseded_at IS NULL")


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
        back_populates="document", passive_deletes=True, order_by="Chunk.position, Chunk.id"
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
    # When a later ingestion replaced this chunk. Superseded chunks are kept so that questions
    # generated from them keep their exact sources; only current ones are searchable.
    superseded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
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
        # Positions are unique among the current chunks; the superseded ones keep theirs.
        Index(
            "chunks_current_position_idx",
            "document_id",
            "position",
            unique=True,
            postgresql_where=CURRENT_CHUNKS,
        ),
        Index("ix_chunks_document_id", "document_id"),
        # The search indexes cover the current chunks only. Besides keeping them small, this
        # keeps the candidate count honest: pgvector applies a WHERE clause after the index
        # scan, so a full index would return fewer current chunks than were asked for.
        Index(
            "chunks_embedding_idx",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "halfvec_cosine_ops"},
            postgresql_where=CURRENT_CHUNKS,
        ),
        Index(
            "chunks_search_vector_idx",
            "search_vector",
            postgresql_using="gin",
            postgresql_where=CURRENT_CHUNKS,
        ),
    )


class Job(Base):
    """One run of a long job. Postgres doubles as the queue, for ingesting a document and for
    generating a batch of questions alike; a generating job belongs to no document."""

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(Text, server_default="ingest")
    document_id: Mapped[int | None] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[str] = mapped_column(Text, server_default="queued")
    # Parser options, e.g. {"ocr": false, "formulas": true}, or what to generate
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
        _one_of("kind", JOB_KINDS),
        # Keeps "find the oldest queued job of this kind" cheap however many finished jobs
        # pile up. A worker claims one kind, so the kind leads.
        Index("jobs_queued_idx", "kind", "created_at", postgresql_where=text("status = 'queued'")),
    )


class Topic(Base):
    """A cluster of the concept tags that the topic map found, gathered across documents."""

    __tablename__ = "topics"

    id: Mapped[int] = mapped_column(primary_key=True)
    # The tag that stands for the cluster
    name: Mapped[str] = mapped_column(Text, unique=True)
    # Every distinct tag in the cluster, the name included
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text))
    # Centroid of the tag embeddings. There are a few hundred topics at most, so the nearest
    # one is found by scanning them; only the chunks need a vector index.
    embedding: Mapped[list[float]] = mapped_column(HALFVEC(EMBEDDING_DIMENSIONS))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ChunkTopic(Base):
    """Which topics a chunk belongs to, through the tags it was given."""

    __tablename__ = "chunk_topics"

    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    topic_id: Mapped[int] = mapped_column(
        ForeignKey("topics.id", ondelete="CASCADE"), primary_key=True, index=True
    )


class ChunkTags(Base):
    """What the local tagger read out of one chunk, kept so that the topics can be re-clustered
    without running the model over the whole library again."""

    __tablename__ = "chunk_tags"

    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    # One line on what the chunk teaches, or "nothing" for boilerplate
    explains: Mapped[str] = mapped_column(Text)
    # Two to five concept tags
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text))
    # The verdict used when choosing what to ask about: the model's flag unless a rule vetoes it
    worth_asking: Mapped[bool]
    # The model's own flag, kept to show how much the rules add
    model_worth_asking: Mapped[bool]
    # The rule that made the chunk context only, e.g. "code only" or "boilerplate"
    skip_reason: Mapped[str | None] = mapped_column(Text)
    model: Mapped[str] = mapped_column(Text)
    prompt_version: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Question(Base):
    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    reference_answer: Mapped[str] = mapped_column(Text)
    # What an answer has to cover, as [{"text", "weight", "evidence_quote", "chunk_id"}].
    # Grading in Phase 3 scores an answer against these.
    key_points: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, server_default="[]")
    # Wrong answers that are worth recognizing
    misconceptions: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default="{}")
    style: Mapped[str] = mapped_column(Text)
    # 1 (recall) to 5 (reasoning across sources)
    difficulty: Mapped[int]
    topic_id: Mapped[int | None] = mapped_column(
        ForeignKey("topics.id", ondelete="SET NULL"), index=True
    )
    status: Mapped[str] = mapped_column(Text, server_default="accepted")
    # What each validation check found, whether or not the question passed
    validation: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    # The model that wrote the question, e.g. "groq:openai/gpt-oss-120b"
    generator_model: Mapped[str] = mapped_column(Text)
    prompt_version: Mapped[str] = mapped_column(Text)
    # Requests and tokens the question cost
    usage: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    # Compared against the other questions to catch near-duplicates; null while the local
    # embedding model is unavailable
    embedding: Mapped[list[float] | None] = mapped_column(HALFVEC(EMBEDDING_DIMENSIONS))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    topic: Mapped[Topic | None] = relationship()
    sources: Mapped[list["QuestionSource"]] = relationship(
        back_populates="question", passive_deletes=True, order_by="QuestionSource.position"
    )

    __table_args__ = (
        _one_of("status", QUESTION_STATUSES),
        _one_of("style", QUESTION_STYLES),
        CheckConstraint("difficulty BETWEEN 1 AND 5", name="difficulty_valid"),
        Index(
            "questions_embedding_idx",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "halfvec_cosine_ops"},
        ),
    )


class QuestionSource(Base):
    """The chunks a question was written from, in the order the model was shown them.

    The reference restricts deleting a chunk, and with it the document it belongs to, while a
    question still cites it: a question without its sources can neither be graded nor checked.
    """

    __tablename__ = "question_sources"

    question_id: Mapped[int] = mapped_column(
        ForeignKey("questions.id", ondelete="CASCADE"), primary_key=True
    )
    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("chunks.id", ondelete="RESTRICT"), primary_key=True, index=True
    )
    position: Mapped[int]

    question: Mapped[Question] = relationship(back_populates="sources")
    chunk: Mapped[Chunk] = relationship()


def source_updated() -> ColumnElement[bool]:
    """Whether a later ingestion has replaced any of a question's sources.

    The question still points at the exact text it was written from, but the document has moved
    on, so the question is worth reviewing. Derived rather than stored: it cannot fall behind
    the chunks that way.
    """
    return (
        select(QuestionSource.question_id)
        .join(Chunk, Chunk.id == QuestionSource.chunk_id)
        .where(QuestionSource.question_id == Question.id, Chunk.superseded_at.is_not(None))
        .exists()
    )


class QuestionTask(Base):
    """One question a generating job set out to write.

    The plan is written down before any of it runs, so a job that is interrupted -- or that
    runs out of the day's tokens -- picks up where it stopped instead of starting over.
    """

    __tablename__ = "question_tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    # Order within the job
    position: Mapped[int]
    # The chunks to write from, in the order the model is shown them
    chunk_ids: Mapped[list[int]] = mapped_column(ARRAY(BigInteger))
    style: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, server_default="queued")
    # The question that came out, whether it was accepted or rejected
    question_id: Mapped[int | None] = mapped_column(ForeignKey("questions.id", ondelete="SET NULL"))
    error: Mapped[str | None] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        _one_of("status", TASK_STATUSES),
        _one_of("style", QUESTION_STYLES),
        UniqueConstraint("job_id", "position"),
    )

"""Topic map over the chunks, and the generated questions with their sources

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import HALFVEC
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

STYLES = (
    "intuition",
    "why_how",
    "compare",
    "tradeoffs",
    "failure_modes",
    "connection",
    "paper",
)


def created_at() -> sa.Column:
    return sa.Column(
        "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
    )


def upgrade() -> None:
    op.create_table(
        "topics",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False, unique=True),
        sa.Column("tags", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("embedding", HALFVEC(1024), nullable=False),
        created_at(),
    )

    op.create_table(
        "chunk_topics",
        sa.Column(
            "chunk_id",
            sa.BigInteger(),
            sa.ForeignKey("chunks.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "topic_id",
            sa.Integer(),
            sa.ForeignKey("topics.id", ondelete="CASCADE"),
            primary_key=True,
        ),
    )
    op.create_index("ix_chunk_topics_topic_id", "chunk_topics", ["topic_id"])

    op.create_table(
        "chunk_tags",
        sa.Column(
            "chunk_id",
            sa.BigInteger(),
            sa.ForeignKey("chunks.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("explains", sa.Text(), nullable=False),
        sa.Column("tags", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("worth_asking", sa.Boolean(), nullable=False),
        sa.Column("model_worth_asking", sa.Boolean(), nullable=False),
        sa.Column("skip_reason", sa.Text()),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("prompt_version", sa.Text(), nullable=False),
        created_at(),
    )

    op.create_table(
        "questions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("reference_answer", sa.Text(), nullable=False),
        sa.Column("key_points", postgresql.JSONB(), server_default="[]", nullable=False),
        sa.Column(
            "misconceptions", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False
        ),
        sa.Column("style", sa.Text(), nullable=False),
        sa.Column("difficulty", sa.Integer(), nullable=False),
        sa.Column("topic_id", sa.Integer(), sa.ForeignKey("topics.id", ondelete="SET NULL")),
        sa.Column("status", sa.Text(), server_default="accepted", nullable=False),
        sa.Column("validation", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("generator_model", sa.Text(), nullable=False),
        sa.Column("prompt_version", sa.Text(), nullable=False),
        sa.Column("usage", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("embedding", HALFVEC(1024)),
        created_at(),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("status IN ('accepted', 'rejected', 'retired')", name="status_valid"),
        sa.CheckConstraint(
            "style IN (" + ", ".join(f"'{style}'" for style in STYLES) + ")", name="style_valid"
        ),
        sa.CheckConstraint("difficulty BETWEEN 1 AND 5", name="difficulty_valid"),
    )
    op.create_index("ix_questions_topic_id", "questions", ["topic_id"])
    op.create_index(
        "questions_embedding_idx",
        "questions",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "halfvec_cosine_ops"},
    )

    op.create_table(
        "question_sources",
        sa.Column(
            "question_id",
            sa.Integer(),
            sa.ForeignKey("questions.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        # Restricted, not cascaded: a question without its sources can neither be graded nor
        # checked, so a chunk that one cites cannot be deleted out from under it.
        sa.Column(
            "chunk_id",
            sa.BigInteger(),
            sa.ForeignKey("chunks.id", ondelete="RESTRICT"),
            primary_key=True,
        ),
        sa.Column("position", sa.Integer(), nullable=False),
    )
    op.create_index("ix_question_sources_chunk_id", "question_sources", ["chunk_id"])


def downgrade() -> None:
    op.drop_table("question_sources")
    op.drop_table("questions")
    op.drop_table("chunk_tags")
    op.drop_table("chunk_topics")
    op.drop_table("topics")

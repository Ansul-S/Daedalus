"""Mock interviews, and the tables LangGraph keeps their place in

Revision ID: 0012
Revises: 0011
Create Date: 2026-10-09

LangGraph's saver sets up its own tables by running its migrations in turn, the last of them
creating indexes concurrently, which can't run inside a transaction. Its tables are created
here instead, as those migrations leave them in langgraph-checkpoint-postgres 3.1, and its
record of them is written too, so that the saver finds nothing left to do.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

revision: str = "0012"
down_revision: str | Sequence[str] | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# How many migrations langgraph-checkpoint-postgres 3.1 runs: its record counts them from 0
SAVER_MIGRATIONS = 10

SAVER_TABLES = """
CREATE TABLE checkpoint_migrations (
    v INTEGER PRIMARY KEY
);
CREATE TABLE checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);
CREATE TABLE checkpoint_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    version TEXT NOT NULL,
    type TEXT NOT NULL,
    blob BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);
CREATE TABLE checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    blob BYTEA NOT NULL,
    task_path TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);
CREATE INDEX checkpoints_thread_id_idx ON checkpoints(thread_id);
CREATE INDEX checkpoint_blobs_thread_id_idx ON checkpoint_blobs(thread_id);
CREATE INDEX checkpoint_writes_thread_id_idx ON checkpoint_writes(thread_id);
"""


def upgrade() -> None:
    op.create_table(
        "interviews",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("questions", ARRAY(sa.Integer()), nullable=False),
        sa.Column("topic_id", sa.Integer(), sa.ForeignKey("topics.id", ondelete="SET NULL")),
        sa.Column("status", sa.Text(), server_default="asking", nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("moving_since", sa.DateTime(timezone=True)),
        sa.CheckConstraint("status IN ('asking', 'finished', 'ended')", name="status_valid"),
        sa.CheckConstraint("cardinality(questions) BETWEEN 1 AND 5", name="questions_valid"),
    )
    op.create_index("ix_interviews_user_id", "interviews", ["user_id"])
    op.create_table(
        "interview_turns",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "interview_id",
            sa.Integer(),
            sa.ForeignKey("interviews.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("round", sa.Integer(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column(
            "question_id",
            sa.Integer(),
            sa.ForeignKey("questions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("attempt_id", sa.Integer(), sa.ForeignKey("attempts.id", ondelete="SET NULL")),
        sa.Column("no_follow_up", sa.Text()),
        sa.Column("text", sa.Text()),
        sa.Column("key_points", JSONB()),
        sa.Column("chunk_ids", ARRAY(sa.Integer())),
        sa.Column("aim", JSONB()),
        sa.Column("writer_model", sa.Text()),
        sa.Column("prompt_version", sa.Text()),
        sa.Column("usage", JSONB(), server_default="{}", nullable=False),
        sa.Column("answer", sa.Text()),
        sa.Column("seconds", sa.Float()),
        sa.Column("answered_at", sa.DateTime(timezone=True)),
        sa.Column("grade", JSONB()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("kind IN ('question', 'follow_up')", name="kind_valid"),
        sa.UniqueConstraint("interview_id", "round", "kind"),
        sa.CheckConstraint(
            "(kind = 'follow_up') = (text IS NOT NULL) "
            "AND (kind = 'follow_up' OR (answer IS NULL AND grade IS NULL))",
            name="follow_up_fields",
        ),
        sa.CheckConstraint("seconds >= 0", name="seconds_valid"),
    )
    op.create_index("ix_interview_turns_interview_id", "interview_turns", ["interview_id"])
    op.execute(SAVER_TABLES)
    op.execute(
        f"INSERT INTO checkpoint_migrations (v) SELECT generate_series(0, {SAVER_MIGRATIONS - 1})"
    )


def downgrade() -> None:
    for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints", "checkpoint_migrations"):
        op.drop_table(table)
    op.drop_table("interview_turns")
    op.drop_table("interviews")

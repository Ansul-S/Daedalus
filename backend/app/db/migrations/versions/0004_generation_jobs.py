"""Job kinds, and the tasks a question-generating job works through

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
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


def upgrade() -> None:
    op.add_column("jobs", sa.Column("kind", sa.Text(), server_default="ingest", nullable=False))
    op.create_check_constraint("kind_valid", "jobs", "kind IN ('ingest', 'generate')")
    # A job that generates questions belongs to no single document.
    op.alter_column("jobs", "document_id", nullable=True)
    # A worker claims one kind of job, so the kind leads the queue index.
    op.drop_index("jobs_queued_idx", table_name="jobs")
    op.create_index(
        "jobs_queued_idx",
        "jobs",
        ["kind", "created_at"],
        postgresql_where=sa.text("status = 'queued'"),
    )

    op.create_table(
        "question_tasks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "job_id", sa.Integer(), sa.ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("chunk_ids", postgresql.ARRAY(sa.BigInteger()), nullable=False),
        sa.Column("style", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), server_default="queued", nullable=False),
        sa.Column("question_id", sa.Integer(), sa.ForeignKey("questions.id", ondelete="SET NULL")),
        sa.Column("error", sa.Text()),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'done', 'failed')", name="status_valid"
        ),
        sa.CheckConstraint(
            "style IN (" + ", ".join(f"'{style}'" for style in STYLES) + ")", name="style_valid"
        ),
        sa.UniqueConstraint("job_id", "position"),
    )
    op.create_index("ix_question_tasks_job_id", "question_tasks", ["job_id"])


def downgrade() -> None:
    op.drop_table("question_tasks")
    op.drop_index("jobs_queued_idx", table_name="jobs")
    op.create_index(
        "jobs_queued_idx", "jobs", ["created_at"], postgresql_where=sa.text("status = 'queued'")
    )
    # Every remaining job has to belong to a document again.
    op.execute("DELETE FROM jobs WHERE document_id IS NULL")
    op.alter_column("jobs", "document_id", nullable=False)
    op.drop_constraint("kind_valid", "jobs", type_="check")
    op.drop_column("jobs", "kind")

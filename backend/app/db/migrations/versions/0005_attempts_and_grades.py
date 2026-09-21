"""Answers given to questions, and their grades

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-21
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | Sequence[str] | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "attempts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "question_id",
            sa.Integer(),
            sa.ForeignKey("questions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_attempts_question_id", "attempts", ["question_id"])

    op.create_table(
        "grades",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "attempt_id",
            sa.Integer(),
            sa.ForeignKey("attempts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column("grader_model", sa.Text()),
        sa.Column("prompt_version", sa.Text(), nullable=False),
        sa.Column("key_points", postgresql.JSONB(), server_default="[]", nullable=False),
        sa.Column("claims", postgresql.JSONB(), server_default="[]", nullable=False),
        sa.Column("clarity", sa.Integer()),
        sa.Column("strengths", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("gaps", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("errors", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("improved_answer", sa.Text()),
        sa.Column("follow_up", sa.Text()),
        sa.Column("coverage", sa.Float()),
        sa.Column("contradicted", sa.Integer()),
        sa.Column("score", sa.Float()),
        sa.Column("usage", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("seconds", sa.Float()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("status IN ('graded', 'failed')", name="status_valid"),
        sa.CheckConstraint("clarity BETWEEN 1 AND 5", name="clarity_valid"),
        sa.CheckConstraint(
            "score BETWEEN 0 AND 1 AND coverage BETWEEN 0 AND 1", name="score_valid"
        ),
        sa.CheckConstraint(
            "(status = 'graded' AND score IS NOT NULL AND coverage IS NOT NULL "
            "AND contradicted IS NOT NULL AND clarity IS NOT NULL AND grader_model IS NOT NULL) "
            "OR (status = 'failed' AND error IS NOT NULL AND score IS NULL)",
            name="graded_or_failed",
        ),
    )
    op.create_index("ix_grades_attempt_id", "grades", ["attempt_id"])


def downgrade() -> None:
    op.drop_table("grades")
    op.drop_table("attempts")

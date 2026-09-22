"""The review schedule, and how long an answer took

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006"
down_revision: str | Sequence[str] | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("attempts", sa.Column("seconds", sa.Float()))
    op.add_column("attempts", sa.Column("time_limit", sa.Integer()))
    op.create_check_constraint("seconds_valid", "attempts", "seconds >= 0")
    op.create_check_constraint("time_limit_valid", "attempts", "time_limit > 0")

    op.create_table(
        "cards",
        sa.Column(
            "question_id",
            sa.Integer(),
            sa.ForeignKey("questions.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("state", postgresql.JSONB(), nullable=False),
        sa.Column("due", sa.Date(), nullable=False),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_cards_due", "cards", ["due"])

    op.create_table(
        "reviews",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "question_id",
            sa.Integer(),
            sa.ForeignKey("questions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "attempt_id",
            sa.Integer(),
            sa.ForeignKey("attempts.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "grade_id", sa.Integer(), sa.ForeignKey("grades.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("rating", sa.SmallInteger(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("due", sa.Date(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("rating BETWEEN 1 AND 4", name="rating_valid"),
        sa.CheckConstraint("score BETWEEN 0 AND 1", name="score_valid"),
        sa.CheckConstraint("due > day", name="due_valid"),
    )
    op.create_index("ix_reviews_question_id", "reviews", ["question_id"])


def downgrade() -> None:
    op.drop_table("reviews")
    op.drop_table("cards")
    # Dropping a column drops the check on it too.
    op.drop_column("attempts", "time_limit")
    op.drop_column("attempts", "seconds")

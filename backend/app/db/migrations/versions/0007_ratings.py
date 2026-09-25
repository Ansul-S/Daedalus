"""Ratings of questions and grades

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | Sequence[str] | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ratings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("question_id", sa.Integer(), sa.ForeignKey("questions.id", ondelete="CASCADE")),
        sa.Column("grade_id", sa.Integer(), sa.ForeignKey("grades.id", ondelete="CASCADE")),
        sa.Column("value", sa.SmallInteger(), nullable=False),
        sa.Column("note", sa.Text()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("value IN (-1, 1)", name="value_valid"),
        sa.CheckConstraint("(question_id IS NULL) <> (grade_id IS NULL)", name="question_or_grade"),
    )
    op.create_index("ix_ratings_question_id", "ratings", ["question_id"])
    op.create_index("ix_ratings_grade_id", "ratings", ["grade_id"])


def downgrade() -> None:
    op.drop_table("ratings")

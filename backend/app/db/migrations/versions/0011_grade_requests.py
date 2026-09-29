"""The grades asked of a model, for the daily limits

Revision ID: 0011
Revises: 0010
Create Date: 2026-09-29

The daily limits and the pacer count grades here rather than in `grades`, which lose their rows
when practice is deleted. The graded answers of the last two days, as long as the table keeps
anything, are brought over so that today's count carries on.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0011"
down_revision: str | Sequence[str] | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "grade_requests",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("grade_id", sa.Integer(), sa.ForeignKey("grades.id", ondelete="SET NULL")),
        sa.Column("model", sa.Text()),
        sa.Column("requests", sa.Integer(), server_default="0", nullable=False),
        sa.Column("input_tokens", sa.Integer(), server_default="0", nullable=False),
        sa.Column("output_tokens", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_grade_requests_created_at", "grade_requests", ["created_at"])
    op.create_index(
        "ix_grade_requests_user_id_created_at", "grade_requests", ["user_id", "created_at"]
    )
    # Failed grades recorded no usage until now, so only the graded ones can be counted.
    op.execute(
        """
        INSERT INTO grade_requests
            (user_id, grade_id, model, requests, input_tokens, output_tokens, created_at)
        SELECT attempts.user_id, grades.id, grades.grader_model,
               COALESCE((grades.usage ->> 'requests')::int, 1),
               COALESCE((grades.usage ->> 'input_tokens')::int, 0),
               COALESCE((grades.usage ->> 'output_tokens')::int, 0),
               grades.created_at
        FROM grades JOIN attempts ON attempts.id = grades.attempt_id
        WHERE grades.status = 'graded' AND grades.created_at > now() - interval '2 days'
        ORDER BY grades.id
        """
    )


def downgrade() -> None:
    op.drop_table("grade_requests")

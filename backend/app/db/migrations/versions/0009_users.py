"""Users, and whose practice is whose

Revision ID: 0009
Revises: 0008
Create Date: 2026-09-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009"
down_revision: str | Sequence[str] | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Practice: what a user owns. Grades follow their attempt.
OWNED = ("attempts", "reviews", "cards", "ratings")
# The built-in user, `LOCAL_USER` in the models
LOCAL = "provider = 'local' AND subject = 'local'"


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("subject", sa.Text(), nullable=False),
        sa.Column("name", sa.Text()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("provider", "subject"),
    )
    # Everything practised so far was practised locally, by the built-in user.
    op.execute("INSERT INTO users (provider, subject) VALUES ('local', 'local')")
    for table in OWNED:
        op.add_column(
            table,
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE")),
        )
        op.execute(f"UPDATE {table} SET user_id = (SELECT id FROM users WHERE {LOCAL})")
        op.alter_column(table, "user_id", nullable=False)
    for table in ("attempts", "reviews", "ratings"):
        op.create_index(f"ix_{table}_user_id", table, ["user_id"])
    # A card is now one user's place in the schedule for a question.
    op.drop_constraint("cards_pkey", "cards", type_="primary")
    op.create_primary_key("cards_pkey", "cards", ["user_id", "question_id"])
    op.create_index("ix_cards_question_id", "cards", ["question_id"])
    op.drop_index("ix_cards_due", "cards")
    op.create_index("ix_cards_user_id_due", "cards", ["user_id", "due"])


def downgrade() -> None:
    # Without users there is one person's practice: the built-in user's is kept, and everyone
    # else's is removed (their grades and reviews go with their attempts).
    for table in ("attempts", "cards", "ratings"):
        op.execute(f"DELETE FROM {table} WHERE user_id NOT IN (SELECT id FROM users WHERE {LOCAL})")
    op.drop_index("ix_cards_user_id_due", "cards")
    op.create_index("ix_cards_due", "cards", ["due"])
    op.drop_index("ix_cards_question_id", "cards")
    op.drop_constraint("cards_pkey", "cards", type_="primary")
    op.create_primary_key("cards_pkey", "cards", ["question_id"])
    # Dropping a column drops its index and foreign key too.
    for table in OWNED:
        op.drop_column(table, "user_id")
    op.drop_table("users")

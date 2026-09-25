"""Building the topic map as a queued job

Revision ID: 0008
Revises: 0007
Create Date: 2026-09-25
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0008"
down_revision: str | Sequence[str] | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("kind_valid", "jobs", type_="check")
    op.create_check_constraint("kind_valid", "jobs", "kind IN ('ingest', 'generate', 'topics')")


def downgrade() -> None:
    # A topic map job leaves nothing behind but its row: the tags and topics it wrote stay.
    op.execute("DELETE FROM jobs WHERE kind = 'topics'")
    op.drop_constraint("kind_valid", "jobs", type_="check")
    op.create_check_constraint("kind_valid", "jobs", "kind IN ('ingest', 'generate')")

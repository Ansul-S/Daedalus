"""Keep the chunks of earlier ingestions instead of deleting them

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

CURRENT = sa.text("superseded_at IS NULL")


def upgrade() -> None:
    op.add_column("chunks", sa.Column("superseded_at", sa.DateTime(timezone=True)))

    # Superseded chunks keep the position they were ingested at, so only the current chunks
    # of a document can be required to have distinct positions.
    op.drop_constraint("chunks_document_id_position_key", "chunks", type_="unique")
    op.create_index(
        "chunks_current_position_idx",
        "chunks",
        ["document_id", "position"],
        unique=True,
        postgresql_where=CURRENT,
    )
    # That constraint also indexed the foreign key, which deleting a document cascades over.
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])

    # Search reads current chunks only, so its indexes cover only those.
    op.drop_index("chunks_embedding_idx", table_name="chunks")
    op.create_index(
        "chunks_embedding_idx",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "halfvec_cosine_ops"},
        postgresql_where=CURRENT,
    )
    op.drop_index("chunks_search_vector_idx", table_name="chunks")
    op.create_index(
        "chunks_search_vector_idx",
        "chunks",
        ["search_vector"],
        postgresql_using="gin",
        postgresql_where=CURRENT,
    )


def downgrade() -> None:
    op.drop_index("chunks_search_vector_idx", table_name="chunks")
    op.create_index("chunks_search_vector_idx", "chunks", ["search_vector"], postgresql_using="gin")
    op.drop_index("chunks_embedding_idx", table_name="chunks")
    op.create_index(
        "chunks_embedding_idx",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "halfvec_cosine_ops"},
    )
    op.drop_index("ix_chunks_document_id", table_name="chunks")
    op.drop_index("chunks_current_position_idx", table_name="chunks")
    # Without the column a document can only hold one generation of chunks.
    op.execute("DELETE FROM chunks WHERE superseded_at IS NOT NULL")
    op.create_unique_constraint(
        "chunks_document_id_position_key", "chunks", ["document_id", "position"]
    )
    op.drop_column("chunks", "superseded_at")

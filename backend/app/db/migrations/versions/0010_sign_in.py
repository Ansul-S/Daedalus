"""Sign-in: Better Auth's tables

Revision ID: 0010
Revises: 0009
Create Date: 2026-09-29

Better Auth, in the frontend, signs visitors in with GitHub and keeps its users, sessions and
signing keys here. The tables are laid out as Better Auth 1.7's own migration generator writes
them for Postgres, with the table names prefixed `auth_`.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010"
down_revision: str | Sequence[str] | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def stamp(name: str) -> sa.Column:
    return sa.Column(
        name,
        sa.DateTime(timezone=True),
        server_default=sa.text("CURRENT_TIMESTAMP"),
        nullable=False,
    )


def owner() -> sa.Column:
    return sa.Column(
        "userId",
        sa.Text(),
        sa.ForeignKey("auth_user.id", ondelete="CASCADE"),
        nullable=False,
    )


def upgrade() -> None:
    op.create_table(
        "auth_user",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("email", sa.Text(), nullable=False, unique=True),
        sa.Column("emailVerified", sa.Boolean(), nullable=False),
        sa.Column("image", sa.Text()),
        stamp("createdAt"),
        stamp("updatedAt"),
    )
    op.create_table(
        "auth_session",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("expiresAt", sa.DateTime(timezone=True), nullable=False),
        sa.Column("token", sa.Text(), nullable=False, unique=True),
        stamp("createdAt"),
        sa.Column("updatedAt", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ipAddress", sa.Text()),
        sa.Column("userAgent", sa.Text()),
        owner(),
    )
    op.create_index("auth_session_userId_idx", "auth_session", ["userId"])
    op.create_table(
        "auth_account",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("accountId", sa.Text(), nullable=False),
        sa.Column("providerId", sa.Text(), nullable=False),
        owner(),
        sa.Column("accessToken", sa.Text()),
        sa.Column("refreshToken", sa.Text()),
        sa.Column("idToken", sa.Text()),
        sa.Column("accessTokenExpiresAt", sa.DateTime(timezone=True)),
        sa.Column("refreshTokenExpiresAt", sa.DateTime(timezone=True)),
        sa.Column("scope", sa.Text()),
        sa.Column("password", sa.Text()),
        stamp("createdAt"),
        sa.Column("updatedAt", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("auth_account_userId_idx", "auth_account", ["userId"])
    op.create_table(
        "auth_verification",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("identifier", sa.Text(), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("expiresAt", sa.DateTime(timezone=True), nullable=False),
        stamp("createdAt"),
        stamp("updatedAt"),
    )
    op.create_index("auth_verification_identifier_idx", "auth_verification", ["identifier"])
    op.create_table(
        "auth_jwks",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("publicKey", sa.Text(), nullable=False),
        sa.Column("privateKey", sa.Text(), nullable=False),
        sa.Column("createdAt", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expiresAt", sa.DateTime(timezone=True)),
        sa.Column("alg", sa.Text()),
        sa.Column("crv", sa.Text()),
    )


def downgrade() -> None:
    # Everyone is signed out. The API's own users and their practice stay.
    for table in ("auth_jwks", "auth_verification", "auth_account", "auth_session", "auth_user"):
        op.drop_table(table)

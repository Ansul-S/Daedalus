"""Fixtures for tests that need a real PostgreSQL database.

A throwaway database is created once per session, migrated, and dropped
afterwards. Tests are skipped rather than failed when no server is reachable,
so a checkout without PostgreSQL still runs the rest of the suite.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from daedalus.storage.database import DATABASE_URL_ENV

TEST_DATABASE = "daedalus_test"
MIGRATIONS = Path(__file__).resolve().parents[1] / "migrations"


def _url_for(database: str) -> str:
    """Return the configured connection string pointed at another database."""
    parts = conninfo_to_dict(os.environ[DATABASE_URL_ENV])
    parts["dbname"] = database
    return make_conninfo(**parts)


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    """Create, migrate, and finally drop a throwaway database."""
    if not os.environ.get(DATABASE_URL_ENV, "").strip():
        pytest.skip(f"{DATABASE_URL_ENV} is not set")

    try:
        admin = psycopg.connect(_url_for("postgres"), autocommit=True)
    except psycopg.OperationalError as error:
        pytest.skip(f"no PostgreSQL server reachable: {error}")

    name = sql.Identifier(TEST_DATABASE)
    with admin:
        admin.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(name))
        admin.execute(sql.SQL("CREATE DATABASE {}").format(name))

        url = _url_for(TEST_DATABASE)
        with psycopg.connect(url) as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            for migration in sorted(MIGRATIONS.glob("*.sql")):
                connection.execute(migration.read_text())

        yield url

        admin.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(name))


@pytest.fixture
def connection(
    database_url: str,
) -> Iterator[psycopg.Connection[tuple[object, ...]]]:
    """Yield a connection to an empty database, rolled back afterwards."""
    with psycopg.connect(database_url) as conn:
        conn.execute(
            "TRUNCATE documents, chunks, embeddings, queries, judgements "
            "RESTART IDENTITY CASCADE"
        )
        conn.commit()
        yield conn

"""Connection handling for the PostgreSQL store.

The connection string is required rather than defaulted. A default would let
the application silently connect to whichever server happened to answer, which
is the failure mode this project has already hit once.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg

#: Environment variable holding the libpq connection string.
DATABASE_URL_ENV = "DAEDALUS_DATABASE_URL"


class DatabaseNotConfiguredError(RuntimeError):
    """Raised when the connection string is absent from the environment."""


def database_url() -> str:
    """Return the configured connection string.

    Raises DatabaseNotConfiguredError if the environment variable is unset or
    empty, naming the variable so the fix is obvious.
    """
    url = os.environ.get(DATABASE_URL_ENV, "").strip()
    if not url:
        raise DatabaseNotConfiguredError(
            f"{DATABASE_URL_ENV} is not set. Example: "
            f"postgresql:///daedalus?host=/tmp&port=5434"
        )
    return url


@contextmanager
def connect(url: str | None = None) -> Iterator[psycopg.Connection[tuple[object, ...]]]:
    """Open a connection, committing on success and rolling back on error.

    The url argument exists so tests can target a throwaway database without
    changing the environment.
    """
    with psycopg.connect(url or database_url()) as connection:
        yield connection

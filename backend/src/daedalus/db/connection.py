"""
Database connections.

Every connection needs the same setup before it is usable: the sqlite-vec
extension loaded, WAL enabled, and foreign keys turned on. Getting one of
these wrong produces failures that look unrelated to their cause, so they
live in exactly one place.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import sqlite_vec

from daedalus.config import constants

__all__ = ["connect", "get_connection"]


logger = logging.getLogger(__name__)


def _configure(connection: sqlite3.Connection) -> None:
    """Apply the pragmas and extensions every connection depends on."""

    # sqlite-vec ships as a loadable extension. Extension loading is disabled
    # again immediately afterwards: it is a code-execution surface, and
    # nothing else in the application needs it.
    connection.enable_load_extension(True)
    sqlite_vec.load(connection)
    connection.enable_load_extension(False)

    # WAL lets readers proceed while a document is being ingested. Without
    # it, SQLite's single writer blocks every query during indexing.
    connection.execute("PRAGMA journal_mode = WAL")

    # Off by default in SQLite, so ON DELETE CASCADE on chunks.doc_id would
    # silently do nothing.
    connection.execute("PRAGMA foreign_keys = ON")

    # Wait rather than fail immediately when the single writer is busy.
    connection.execute("PRAGMA busy_timeout = 5000")

    connection.row_factory = sqlite3.Row


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    """
    Open a configured connection.

    Passing ``":memory:"`` gives an isolated in-memory database, which is
    what the tests use.
    """

    target = constants.DB_PATH if path is None else path

    if isinstance(target, Path):
        target.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(target)
    _configure(connection)

    return connection


@contextmanager
def get_connection(path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    """Open a configured connection and guarantee it is closed."""

    connection = connect(path)
    try:
        yield connection
    finally:
        connection.close()

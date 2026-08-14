"""Fixtures for the generation tests."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pytest

from daedalus.db import connect, initialize_schema


@pytest.fixture
def db() -> Iterator[sqlite3.Connection]:
    connection = connect(":memory:")
    initialize_schema(connection)
    try:
        yield connection
    finally:
        connection.close()

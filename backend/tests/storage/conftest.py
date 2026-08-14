"""Fixtures for the storage tests."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pytest

from daedalus.db import connect, initialize_schema
from daedalus.storage import documents


@pytest.fixture
def db() -> Iterator[sqlite3.Connection]:
    connection = connect(":memory:")
    initialize_schema(connection)
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture
def doc_id(db: sqlite3.Connection) -> str:
    """A registered document for chunks to hang off."""

    record = documents.create(
        db,
        doc_id="attention-is-all-you-need",
        filename="attention.pdf",
        source_type="arxiv",
        content_hash="hash-attention",
    )

    return record.id

"""
Database schema.

One SQLite file holds everything: document records, chunks, the dense
vector index, and the lexical index. See ADR-007 for why.

The three chunk tables must stay in lockstep — ``chunks`` is the source of
truth, ``chunks_fts`` mirrors its text for BM25, and ``chunks_vec`` holds
the embeddings. FTS5 is kept in sync by triggers; vectors are written by
the indexing code, since they cannot be derived in SQL.
"""

from __future__ import annotations

import sqlite3

from daedalus.config import constants

__all__ = ["SCHEMA_VERSION", "initialize_schema"]


SCHEMA_VERSION = 2


# Core Tables


_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    source_type   TEXT NOT NULL,
    content_hash  TEXT NOT NULL UNIQUE,
    status        TEXT NOT NULL CHECK (
                      status IN ('pending', 'processing', 'completed', 'failed')
                  ),
    error         TEXT,
    n_pages       INTEGER,
    n_chunks      INTEGER,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
"""

# source_start/source_end are offsets into the document's parsed text, not
# into the original file. They are what lets evaluation labels survive a
# change of chunking strategy - see docs/pipelines/EVALUATION_ENGINE.md.
_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY,
    doc_id        TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal       INTEGER NOT NULL,
    text          TEXT NOT NULL,
    source_start  INTEGER NOT NULL,
    source_end    INTEGER NOT NULL,
    extraction    TEXT NOT NULL,
    page          INTEGER,
    UNIQUE (doc_id, ordinal),
    CHECK (source_end > source_start)
);
"""

_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_extraction ON chunks (extraction);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents (status);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents (content_hash);
"""


# Search Indexes


# External-content FTS5: the index stores only the inverted terms and reads
# document text back from `chunks`, so the text is not duplicated on disk.
_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id',
    tokenize='porter unicode61'
);
"""

# External-content tables do not update themselves. Without these triggers
# the lexical half of hybrid retrieval silently returns stale results.
_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts (rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts (chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts (chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts (rowid, text) VALUES (new.id, new.text);
END;
"""

_VEC = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding FLOAT[{constants.EMBEDDING_DIM}]
);
"""

# Vectors cannot be computed in SQL, so inserts are the indexing code's job.
# Deletes can be, and a foreign key would not help here: ON DELETE CASCADE
# does not reach into a virtual table. Without this trigger, deleting a
# document leaves its vectors behind and dense retrieval keeps returning
# chunk ids that no longer exist.
#
# There is deliberately no update trigger. A chunk's text and its vector
# cannot be kept in step by SQL, so chunks are replaced rather than updated.
_VEC_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS chunks_vec_delete AFTER DELETE ON chunks BEGIN
    DELETE FROM chunks_vec WHERE chunk_id = old.id;
END;
"""


# Initialization


def initialize_schema(connection: sqlite3.Connection) -> None:
    """
    Create every table, index, and trigger if absent.

    Idempotent: safe to call on every startup.
    """

    with connection:
        connection.executescript(_DOCUMENTS)
        connection.executescript(_CHUNKS)
        connection.executescript(_INDEXES)
        connection.executescript(_FTS)
        connection.executescript(_FTS_TRIGGERS)
        connection.executescript(_VEC)
        connection.executescript(_VEC_TRIGGERS)
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

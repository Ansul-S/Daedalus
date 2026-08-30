-- Initial schema: documents, their chunks, and embeddings of those chunks.

CREATE TABLE documents (
    doc_id        text PRIMARY KEY,
    source_path   text NOT NULL,
    source_format text NOT NULL,
    title         text,
    ingested_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
    id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    doc_id         text NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    ordinal        integer NOT NULL,
    kind           text NOT NULL CHECK (kind IN ('prose', 'code', 'output')),
    text           text NOT NULL,
    heading_path   text[] NOT NULL DEFAULT '{}',
    tags           text[] NOT NULL DEFAULT '{}',
    locator        text NOT NULL,
    parent_ordinal integer,

    UNIQUE (doc_id, ordinal),

    -- An output chunk points at the code chunk in the same document that
    -- produced it. Enforced, so an orphaned output cannot be inserted.
    FOREIGN KEY (doc_id, parent_ordinal)
        REFERENCES chunks (doc_id, ordinal) ON DELETE CASCADE,

    -- Only outputs may have a parent.
    CHECK (parent_ordinal IS NULL OR kind = 'output')
);

CREATE TABLE embeddings (
    chunk_id   bigint NOT NULL REFERENCES chunks (id) ON DELETE CASCADE,
    model      text NOT NULL,
    dim        integer NOT NULL,
    embedding  vector NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),

    PRIMARY KEY (chunk_id, model),

    -- The recorded dimension must match the stored vector.
    CHECK (vector_dims(embedding) = dim)
);

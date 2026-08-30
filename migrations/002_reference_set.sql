-- Full-text search over chunks, and the hand-labelled reference set used to
-- measure retrieval quality.

-- A stored tsvector so lexical search does not recompute it per row per query.
-- The expression covers the chunk text only: array_to_string is STABLE rather
-- than IMMUTABLE and so cannot appear in a generated column. The lexical view
-- therefore differs from the embedding input, which does include the heading
-- path. That difference is wanted: pooling draws candidates from retrievers
-- that see the corpus differently.
ALTER TABLE chunks ADD COLUMN search_vector tsvector
    GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;

CREATE TABLE queries (
    id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    text       text NOT NULL UNIQUE,
    source     text NOT NULL CHECK (source IN ('harvested', 'authored')),
    created_at timestamptz NOT NULL DEFAULT now()
);

-- Judgements are keyed by (doc_id, ordinal) and carry NO foreign key into
-- documents or chunks. This is deliberate, and it is a trade.
--
-- Storing a document deletes and reinserts its rows, so any foreign key into
-- documents or chunks would cascade and destroy judgements every time material
-- was re-ingested — discarding hours of human labelling to save milliseconds of
-- parsing. Because doc_id is a content hash, (doc_id, ordinal) identifies the
-- same content across any number of re-ingests.
--
-- The cost is that a judgement can outlive the content it describes: if a
-- document is edited, its doc_id changes and the old judgements become
-- orphans. They are not deleted automatically; orphaned_judgements() reports
-- them so the decision to discard is explicit rather than silent.
CREATE TABLE judgements (
    query_id  bigint NOT NULL REFERENCES queries (id) ON DELETE CASCADE,
    doc_id    text NOT NULL,
    ordinal   integer NOT NULL,
    grade     smallint NOT NULL CHECK (grade IN (0, 1, 2)),
    judged_at timestamptz NOT NULL DEFAULT now(),

    PRIMARY KEY (query_id, doc_id, ordinal)
);

-- Judgements whose chunk no longer exists, for explicit review.
CREATE VIEW orphaned_judgements AS
SELECT j.*
FROM judgements j
LEFT JOIN chunks c ON c.doc_id = j.doc_id AND c.ordinal = j.ordinal
WHERE c.id IS NULL;

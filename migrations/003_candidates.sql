-- The persisted candidate pool: every chunk ever offered for judgement, and
-- which retriever surfaced it.
--
-- Until now the pool was implicit. It was whatever pool_candidates() recomputed
-- from the vector, lexical and random retrievers, which was reproducible
-- because that draw is deterministic. Adding candidates from retrievers that
-- did not build the original pool ends that: the judged set can no longer be
-- derived from the three original retrievers alone, so it has to be recorded.
--
-- One row per (candidate, source). A chunk surfaced by three retrievers is
-- three rows, so provenance survives rather than collapsing to a single fact
-- that something was pooled.
CREATE TABLE candidates (
    query_id bigint  NOT NULL REFERENCES queries (id) ON DELETE CASCADE,
    doc_id   text    NOT NULL,
    ordinal  integer NOT NULL,
    source   text    NOT NULL,
    rank     integer,
    added_at timestamptz NOT NULL DEFAULT now(),

    PRIMARY KEY (query_id, doc_id, ordinal, source)
);

-- Keyed by (doc_id, ordinal) with no foreign key into chunks, for the reason
-- given on judgements in 002: re-ingesting a document must not cascade.
--
-- There is deliberately no foreign key between candidates and judgements in
-- either direction. A candidate without a judgement is the labelling backlog.
-- A judgement without a candidate row must remain readable, because making the
-- reference set depend on this table would let a mistake here destroy it.

-- Candidates still awaiting a judgement. Completeness is this view being
-- empty; before it existed the only way to check was to recompute every pool.
CREATE VIEW unjudged_candidates AS
SELECT DISTINCT c.query_id, c.doc_id, c.ordinal
FROM candidates c
LEFT JOIN judgements j
       ON j.query_id = c.query_id
      AND j.doc_id   = c.doc_id
      AND j.ordinal  = c.ordinal
WHERE j.query_id IS NULL;

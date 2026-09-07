-- Responses that failed a deterministic check, and were therefore discarded.
--
-- These have to be persisted, not merely reported. docs/PHASE-6-PROTOCOL.md
-- section 9 forbids repairing, re-prompting or regenerating a rejected
-- response; without a record of what was already attempted, rerunning the
-- generator would silently retry every rejected section and turn a measured
-- rejection rate into a best-of-N result.
--
-- Transport failures are deliberately NOT recorded here. A section the model
-- was never reached for has not been generated from, so retrying it completes
-- the run rather than regenerating it.
CREATE TABLE question_rejections (
    doc_id         text    NOT NULL,
    heading_path   text[]  NOT NULL,
    selection_rank integer NOT NULL,
    reason         text    NOT NULL,
    detail         text    NOT NULL,
    raw_response   text,
    model          text    NOT NULL,
    prompt_version text    NOT NULL,
    params_hash    text    NOT NULL,
    rejected_at    timestamptz NOT NULL DEFAULT now(),

    -- One outcome per section, mirroring the one question per section rule.
    PRIMARY KEY (doc_id, heading_path)
);

-- Keyed by (doc_id, heading_path) with no foreign key into chunks or
-- documents, for the reason given in 002, 003 and 004: a re-ingest must not
-- cascade away the record of what was already attempted.

-- The raw response is kept so a rejection can be inspected afterwards. It is
-- the evidence for the rejection reason; discarding it would leave the counts
-- unauditable.

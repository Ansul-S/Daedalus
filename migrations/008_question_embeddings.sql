-- Question embeddings and duplicate-pair labels, for the safety net in
-- docs/PHASE-6-PROTOCOL.md section 11.
--
-- Question vectors cannot live in `embeddings`: that table is keyed
-- (chunk_id, model) with a foreign key into chunks, and widening it to hold two
-- different kinds of thing would put a nullable key on the table every retrieval
-- measurement reads. A separate table keeps the production embeddings untouched,
-- which section 11 step 1 requires.
--
-- The model name carries its own identity, `bge-m3-questions` rather than
-- `bge-m3`. The separate table already prevents collision; the distinct name
-- also makes it impossible to pool question vectors with chunk vectors by
-- writing a query that forgets which is which.
CREATE TABLE question_embeddings (
    question_id bigint      NOT NULL REFERENCES questions (id) ON DELETE CASCADE,
    model       text        NOT NULL,
    dim         integer     NOT NULL,
    embedding   vector      NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),

    PRIMARY KEY (question_id, model),
    CHECK (vector_dims(embedding) = dim)
);

-- One labelled pair. Similarity and band are stored rather than recomputed:
-- the sample is drawn once and frozen, and the analysis must reflect the pair as
-- it was presented. Recomputing later against regenerated vectors would silently
-- re-band a pair the labeller judged under different numbers.
--
-- The ordering constraint makes an unordered pair storable exactly once, so the
-- same two questions cannot be labelled twice under opposite orderings and then
-- disagree with themselves.
CREATE TABLE duplicate_labels (
    lower_id    bigint           NOT NULL REFERENCES questions (id) ON DELETE CASCADE,
    higher_id   bigint           NOT NULL REFERENCES questions (id) ON DELETE CASCADE,
    model       text             NOT NULL,
    similarity  double precision NOT NULL,
    band        text             NOT NULL,
    value       text             NOT NULL CHECK (value IN ('duplicate', 'not_duplicate')),
    labelled_at timestamptz      NOT NULL DEFAULT now(),

    PRIMARY KEY (lower_id, higher_id, model),
    CHECK (lower_id < higher_id),
    CHECK (similarity >= -1.0 AND similarity <= 1.0)
);

-- Generated questions, the chunks they were built from and cite, and the two
-- separate records of how they were assessed.
--
-- The protocol these tables serve is docs/PHASE-6-PROTOCOL.md, committed before
-- any question was generated.

CREATE TABLE questions (
    id                   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    doc_id               text NOT NULL,
    heading_path         text[] NOT NULL,
    seed_ordinal         integer NOT NULL,
    selection_rank       integer NOT NULL,
    requested_type       text NOT NULL CHECK (requested_type IN
                             ('conceptual', 'explanation',
                              'comparison', 'code_reasoning')),
    requested_difficulty text NOT NULL CHECK (requested_difficulty IN
                             ('easy', 'medium', 'hard')),
    text                 text NOT NULL CHECK (btrim(text) <> ''),
    grounding_quote      text NOT NULL CHECK (btrim(grounding_quote) <> ''),
    model                text NOT NULL,
    prompt_version       text NOT NULL,
    params_hash          text NOT NULL,
    generated_at         timestamptz NOT NULL DEFAULT now(),

    -- One question per section, enforced rather than merely intended. The
    -- protocol makes coverage structural by generating from each selected
    -- section exactly once; a second question for a section would silently
    -- break that guarantee and inflate the duplicate rate.
    UNIQUE (doc_id, heading_path),

    -- The position in the frozen selection order. Unique so that two sections
    -- cannot claim the same draw.
    UNIQUE (selection_rank)
);

-- Every chunk supplied as context for a question, and whether the model cited
-- it. Context and citation are different facts: the context is what the
-- generator was shown, the citation is what it claimed to have used. Keeping
-- both is what makes an ungrounded citation detectable.
CREATE TABLE question_sources (
    question_id bigint  NOT NULL REFERENCES questions (id) ON DELETE CASCADE,
    doc_id      text    NOT NULL,
    ordinal     integer NOT NULL,
    role        text    NOT NULL CHECK (role IN ('seed', 'context')),
    cited       boolean NOT NULL,

    PRIMARY KEY (question_id, doc_id, ordinal)
);

-- Keyed by (doc_id, ordinal) with no foreign key into chunks, for the reason
-- given on judgements in 002 and candidates in 003: storing a document deletes
-- and reinserts its rows, so a foreign key would cascade and destroy generated
-- questions and the human labelling attached to them on every re-ingest.

-- Exactly one seed per question. The protocol designates a single focal chunk
-- and requires every question to cite it; two seeds would make "the seed"
-- ambiguous in a way nothing downstream could resolve.
CREATE UNIQUE INDEX one_seed_per_question
    ON question_sources (question_id) WHERE role = 'seed';

-- Human labels. The rubric names are fixed by docs/PHASE-6-PROTOCOL.md section
-- 12; their levels are not, because the rubrics themselves are written later in
-- the phase and must not be pre-empted by a guess made here. A CHECK on value
-- belongs in a later migration, once the rubrics are frozen.
CREATE TABLE question_labels (
    question_id bigint NOT NULL REFERENCES questions (id) ON DELETE CASCADE,
    rubric      text   NOT NULL CHECK (rubric IN
                    ('groundedness', 'relevance', 'difficulty')),
    value       text   NOT NULL,
    labelled_at timestamptz NOT NULL DEFAULT now(),

    PRIMARY KEY (question_id, rubric)
);

-- Automated judge output, deliberately in its own table rather than a column or
-- a flag on question_labels. A judge run must not be able to overwrite a human
-- label, and no query should be able to average the two together by accident.
--
-- run exists because docs/PROJECT.md calls for measuring judge consistency
-- across runs, which requires keeping repeated scores of the same question
-- rather than replacing them.
CREATE TABLE judge_scores (
    question_id   bigint   NOT NULL REFERENCES questions (id) ON DELETE CASCADE,
    rubric        text     NOT NULL CHECK (rubric IN
                      ('groundedness', 'relevance', 'difficulty')),
    value         text     NOT NULL,
    judge_model   text     NOT NULL,
    judge_version text     NOT NULL,
    run           smallint NOT NULL DEFAULT 1 CHECK (run >= 1),
    scored_at     timestamptz NOT NULL DEFAULT now(),

    PRIMARY KEY (question_id, rubric, judge_model, judge_version, run)
);

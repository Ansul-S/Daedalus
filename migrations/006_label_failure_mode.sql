-- Which kind of groundedness failure a label records.
--
-- docs/PHASE-6-RUBRICS.md section 3 distinguishes two ways a question can fail
-- groundedness, and requires the labeller to record which applies:
--
--   support failure   the material needed does not exist in the section at all
--   citation failure  the material exists but is not in the chunks cited
--
-- Both reduce the grade identically, so the mode cannot be inferred from the
-- grade. It is a column rather than a fourth rubric value or a string encoded
-- into `value`: a fourth rubric would put a non-rubric in the rubric column and
-- force every rate query to exclude it, and encoding it into `value` would turn
-- "share at groundedness 2" from an equality test into string parsing, which is
-- how a rate quietly comes out wrong a year later.
ALTER TABLE question_labels ADD COLUMN failure_mode text;

ALTER TABLE question_labels ADD CONSTRAINT failure_mode_is_known
    CHECK (failure_mode IS NULL OR failure_mode IN ('support', 'citation'));

-- The mode is required exactly where the rubric defines one, and forbidden
-- everywhere else: groundedness at grade 0 or 1 has a failure mode, groundedness
-- at grade 2 has none, and relevance and difficulty have none by construction.
--
-- Enforced here rather than in application code because a missing mode is
-- unrecoverable after the fact — it would mean asking the labeller to re-judge
-- an item they have already moved past.
ALTER TABLE question_labels ADD CONSTRAINT failure_mode_matches_grade
    CHECK (
        CASE
            WHEN rubric = 'groundedness' AND value IN ('0', '1')
                THEN failure_mode IS NOT NULL
            ELSE failure_mode IS NULL
        END
    );

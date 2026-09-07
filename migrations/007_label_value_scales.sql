-- The rubric scales, now that docs/PHASE-6-RUBRICS.md has frozen them.
--
-- 004 deliberately left `value` unconstrained: the rubrics had not been written,
-- and guessing their levels would have pre-empted that work. They are now
-- frozen, so the scales become a constraint.
--
-- This also closes a hole left by 006. That migration requires a failure_mode
-- exactly when a groundedness label reads '0' or '1'; a mistyped grade such as
-- 'zero' fell to the ELSE branch and was accepted with no mode at all. With the
-- scale constrained, a grade is either on the scale or rejected outright.
ALTER TABLE question_labels ADD CONSTRAINT value_matches_rubric
    CHECK (
        CASE rubric
            WHEN 'groundedness' THEN value IN ('0', '1', '2')
            WHEN 'relevance'    THEN value IN ('0', '1', '2')
            WHEN 'difficulty'   THEN value IN ('easy', 'medium', 'hard', 'unusable')
            ELSE false
        END
    );

-- judge_scores.value is deliberately NOT constrained.
--
-- A human label is entered through a tool and is either on the scale or a
-- mistake worth blocking. A judge score is whatever a language model emitted,
-- and an off-scale answer is a finding about the judge rather than a write to
-- reject: refusing the insert would discard the evidence that the judge failed
-- to follow its own rubric, which is one of the things Phase 6 sets out to
-- measure. Off-scale judge output is reported, not prevented.

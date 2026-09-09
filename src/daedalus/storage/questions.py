"""Generated questions, their sources, and the two records of assessment.

The invariants enforced here come from `docs/PHASE-6-PROTOCOL.md`: one question
per section, a single seed chunk, and a citation set that is a subset of the
supplied context and always includes the seed. They are checked before the
insert so that a violation names what was wrong rather than surfacing as a
constraint error from the database.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import psycopg

Connection = psycopg.Connection[tuple[object, ...]]

#: Question types the Phase 6 generator produces. Assigned deterministically
#: from section properties, never chosen by the model.
QUESTION_TYPES = ("conceptual", "explanation", "comparison", "code_reasoning")

#: Requested difficulty levels, easiest first.
DIFFICULTIES = ("easy", "medium", "hard")

#: The three rubrics named by the protocol, and the scale each one uses. Frozen
#: in docs/PHASE-6-RUBRICS.md and enforced by the database; repeated here so a
#: bad value is named before it reaches a constraint.
RUBRIC_VALUES = {
    "groundedness": ("0", "1", "2"),
    "relevance": ("0", "1", "2"),
    "difficulty": ("easy", "medium", "hard", "unusable"),
}

RUBRICS = tuple(RUBRIC_VALUES)

#: How a question failed groundedness. Support failure means the material needed
#: is absent from the section; citation failure means it exists but was not among
#: the chunks the question cited. See docs/PHASE-6-RUBRICS.md section 3.
FAILURE_MODES = ("support", "citation")

#: Groundedness grades that require a failure mode. A grade 2 question has
#: neither failure, and relevance and difficulty have none by construction.
GROUNDEDNESS_FAILING_GRADES = ("0", "1")

_INSERT_QUESTION = """
INSERT INTO questions
    (doc_id, heading_path, seed_ordinal, selection_rank, requested_type,
     requested_difficulty, text, grounding_quote, model, prompt_version,
     params_hash)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
RETURNING id
"""

_INSERT_SOURCE = """
INSERT INTO question_sources (question_id, doc_id, ordinal, role, cited)
VALUES (%s, %s, %s, %s, %s)
"""

_UPSERT_LABEL = """
INSERT INTO question_labels (question_id, rubric, value, failure_mode)
VALUES (%s, %s, %s, %s)
ON CONFLICT (question_id, rubric) DO UPDATE
SET value = EXCLUDED.value,
    failure_mode = EXCLUDED.failure_mode,
    labelled_at = now()
"""

_INSERT_JUDGE_SCORE = """
INSERT INTO judge_scores
    (question_id, rubric, value, judge_model, judge_version, run)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (question_id, rubric, judge_model, judge_version, run) DO UPDATE
SET value = EXCLUDED.value, scored_at = now()
"""


@dataclass(frozen=True)
class GeneratedQuestion:
    """One question, with the context it came from and the chunks it cites.

    `context_ordinals` is everything the generator was shown, seed included.
    `cited_ordinals` is what it claimed to have used, and must be a subset.
    """

    doc_id: str
    heading_path: tuple[str, ...]
    seed_ordinal: int
    selection_rank: int
    requested_type: str
    requested_difficulty: str
    text: str
    grounding_quote: str
    context_ordinals: tuple[int, ...]
    cited_ordinals: tuple[int, ...]
    model: str
    prompt_version: str
    params_hash: str


@dataclass(frozen=True)
class StoredQuestion:
    """A question as read back, with its sources split out."""

    question_id: int
    doc_id: str
    heading_path: tuple[str, ...]
    seed_ordinal: int
    selection_rank: int
    requested_type: str
    requested_difficulty: str
    text: str
    grounding_quote: str
    context_ordinals: tuple[int, ...]
    cited_ordinals: tuple[int, ...]


def _validate(question: GeneratedQuestion) -> None:
    """Raise ValueError if a protocol invariant does not hold."""
    if question.requested_type not in QUESTION_TYPES:
        raise ValueError(
            f"requested_type must be one of {QUESTION_TYPES}, "
            f"got {question.requested_type!r}"
        )
    if question.requested_difficulty not in DIFFICULTIES:
        raise ValueError(
            f"requested_difficulty must be one of {DIFFICULTIES}, "
            f"got {question.requested_difficulty!r}"
        )
    if not question.text.strip():
        raise ValueError("question text is empty")
    if not question.grounding_quote.strip():
        raise ValueError("grounding_quote is empty")

    context = set(question.context_ordinals)
    cited = set(question.cited_ordinals)

    if len(context) != len(question.context_ordinals):
        raise ValueError("context_ordinals contains a duplicate")
    if len(cited) != len(question.cited_ordinals):
        raise ValueError("cited_ordinals contains a duplicate")
    if question.seed_ordinal not in context:
        raise ValueError(
            f"seed ordinal {question.seed_ordinal} is not among the context"
        )
    if not cited:
        raise ValueError("cited_ordinals is empty")
    if not cited <= context:
        raise ValueError(
            f"cited ordinals {sorted(cited - context)} were not in the context"
        )
    if question.seed_ordinal not in cited:
        raise ValueError(f"seed ordinal {question.seed_ordinal} is not cited")


def record_question(connection: Connection, question: GeneratedQuestion) -> int:
    """Store a question and every chunk it was shown, returning its id.

    The question and its sources are written together. A section already
    holding a question raises, which is the schema enforcing the protocol's
    one-question-per-section rule rather than this function trusting the caller.
    """
    _validate(question)

    cited = set(question.cited_ordinals)
    with connection.cursor() as cursor:
        cursor.execute(
            _INSERT_QUESTION,
            (
                question.doc_id,
                list(question.heading_path),
                question.seed_ordinal,
                question.selection_rank,
                question.requested_type,
                question.requested_difficulty,
                question.text,
                question.grounding_quote,
                question.model,
                question.prompt_version,
                question.params_hash,
            ),
        )
        row = cursor.fetchone()
        if row is None:  # pragma: no cover - RETURNING always yields a row
            raise RuntimeError("insert did not return a question id")
        question_id = cast("int", row[0])

        cursor.executemany(
            _INSERT_SOURCE,
            [
                (
                    question_id,
                    question.doc_id,
                    ordinal,
                    "seed" if ordinal == question.seed_ordinal else "context",
                    ordinal in cited,
                )
                for ordinal in question.context_ordinals
            ],
        )
    return question_id


def list_questions(connection: Connection) -> list[StoredQuestion]:
    """Return every stored question in selection order, sources included."""
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT q.id, q.doc_id, q.heading_path, q.seed_ordinal,
                   q.selection_rank, q.requested_type, q.requested_difficulty,
                   q.text, q.grounding_quote,
                   array_agg(s.ordinal ORDER BY s.ordinal) AS context,
                   array_remove(
                       array_agg(
                           CASE WHEN s.cited THEN s.ordinal END
                           ORDER BY s.ordinal),
                       NULL) AS cited
            FROM questions q
            JOIN question_sources s ON s.question_id = q.id
            GROUP BY q.id
            ORDER BY q.selection_rank
            """
        )
        return [
            StoredQuestion(
                question_id=cast("int", row[0]),
                doc_id=cast("str", row[1]),
                heading_path=tuple(cast("list[str]", row[2])),
                seed_ordinal=cast("int", row[3]),
                selection_rank=cast("int", row[4]),
                requested_type=cast("str", row[5]),
                requested_difficulty=cast("str", row[6]),
                text=cast("str", row[7]),
                grounding_quote=cast("str", row[8]),
                context_ordinals=tuple(cast("list[int]", row[9])),
                cited_ordinals=tuple(cast("list[int]", row[10])),
            )
            for row in cursor.fetchall()
        ]


def record_label(
    connection: Connection,
    question_id: int,
    rubric: str,
    value: str,
    failure_mode: str | None = None,
) -> None:
    """Record a human rubric label, replacing any earlier label for that rubric.

    The value must be on that rubric's frozen scale, and `failure_mode` is
    required exactly where the groundedness rubric defines one
    — grades 0 and 1 — and must be absent everywhere else. The database enforces
    the same rule; it is checked here first so a mistake names itself rather
    than surfacing as a constraint violation.
    """
    if rubric not in RUBRICS:
        raise ValueError(f"rubric must be one of {RUBRICS}, got {rubric!r}")
    if value not in RUBRIC_VALUES[rubric]:
        raise ValueError(
            f"{rubric} value must be one of {RUBRIC_VALUES[rubric]}, got {value!r}"
        )
    if failure_mode is not None and failure_mode not in FAILURE_MODES:
        raise ValueError(
            f"failure_mode must be one of {FAILURE_MODES}, got {failure_mode!r}"
        )

    needs_mode = rubric == "groundedness" and value in GROUNDEDNESS_FAILING_GRADES
    if needs_mode and failure_mode is None:
        raise ValueError(
            f"groundedness grade {value} requires a failure_mode "
            f"(one of {FAILURE_MODES})"
        )
    if not needs_mode and failure_mode is not None:
        raise ValueError(
            f"failure_mode is only recorded for groundedness grades "
            f"{GROUNDEDNESS_FAILING_GRADES}, not {rubric} {value!r}"
        )

    with connection.cursor() as cursor:
        cursor.execute(_UPSERT_LABEL, (question_id, rubric, value, failure_mode))


def failure_mode_counts(connection: Connection) -> dict[str, int]:
    """Return how many groundedness failures were of each mode.

    Reported as a breakdown of the grade 0 and grade 1 items. It changes no
    rate; it says what the failures were made of.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT failure_mode, count(*) FROM question_labels "
            "WHERE failure_mode IS NOT NULL GROUP BY failure_mode"
        )
        return {cast("str", row[0]): cast("int", row[1]) for row in cursor.fetchall()}


def record_judge_score(
    connection: Connection,
    question_id: int,
    rubric: str,
    value: str,
    judge_model: str,
    judge_version: str,
    run: int = 1,
) -> None:
    """Record one automated score, kept separate from the human labels.

    Repeated runs are stored rather than replaced, so consistency across runs
    stays measurable.
    """
    if rubric not in RUBRICS:
        raise ValueError(f"rubric must be one of {RUBRICS}, got {rubric!r}")
    if not value.strip():
        raise ValueError("judge value is empty")
    if run < 1:
        raise ValueError(f"run must be 1 or greater, got {run}")

    with connection.cursor() as cursor:
        cursor.execute(
            _INSERT_JUDGE_SCORE,
            (question_id, rubric, value, judge_model, judge_version, run),
        )


def labels_for(connection: Connection, question_id: int) -> dict[str, str]:
    """Return the human labels recorded for one question, keyed by rubric."""
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT rubric, value FROM question_labels WHERE question_id = %s",
            (question_id,),
        )
        return {cast("str", row[0]): cast("str", row[1]) for row in cursor.fetchall()}


def sections_with_questions(connection: Connection) -> set[tuple[str, tuple[str, ...]]]:
    """Return the sections that already hold a question.

    Structural coverage is this set measured against the frozen selection, and
    a resumed generation run uses it to skip what is already done.
    """
    with connection.cursor() as cursor:
        cursor.execute("SELECT doc_id, heading_path FROM questions")
        return {
            (cast("str", row[0]), tuple(cast("list[str]", row[1])))
            for row in cursor.fetchall()
        }


def question_counts(connection: Connection) -> dict[str, int]:
    """Return how many questions are stored per requested type."""
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT requested_type, count(*) FROM questions GROUP BY requested_type"
        )
        return {cast("str", row[0]): cast("int", row[1]) for row in cursor.fetchall()}


def record_questions(
    connection: Connection, questions: Sequence[GeneratedQuestion]
) -> list[int]:
    """Store several questions, returning their ids in order."""
    return [record_question(connection, question) for question in questions]


_INSERT_REJECTION = """
INSERT INTO question_rejections
    (doc_id, heading_path, selection_rank, reason, detail, raw_response,
     model, prompt_version, params_hash)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (doc_id, heading_path) DO NOTHING
"""


@dataclass(frozen=True)
class RecordedRejection:
    """A response discarded by a deterministic check, and why."""

    doc_id: str
    heading_path: tuple[str, ...]
    selection_rank: int
    reason: str
    detail: str


def record_rejection(
    connection: Connection,
    doc_id: str,
    heading_path: Sequence[str],
    selection_rank: int,
    reason: str,
    detail: str,
    raw_response: str | None,
    model: str,
    prompt_version: str,
    params_hash: str,
) -> bool:
    """Record that a section's response failed a check, returning whether it was new.

    An existing rejection is left untouched rather than replaced. The protocol
    forbids regenerating a rejected section, so the first outcome is the one
    that counts and a later run must not overwrite it.
    """
    if not reason.strip():
        raise ValueError("rejection reason is empty")

    with connection.cursor() as cursor:
        cursor.execute(
            _INSERT_REJECTION,
            (
                doc_id,
                list(heading_path),
                selection_rank,
                reason,
                detail,
                raw_response,
                model,
                prompt_version,
                params_hash,
            ),
        )
        return cursor.rowcount == 1


def rejected_sections(connection: Connection) -> set[tuple[str, tuple[str, ...]]]:
    """Return the sections whose response was rejected.

    Together with sections_with_questions this is what makes a rerun safe: a
    section appearing in either set has had its one attempt.
    """
    with connection.cursor() as cursor:
        cursor.execute("SELECT doc_id, heading_path FROM question_rejections")
        return {
            (cast("str", row[0]), tuple(cast("list[str]", row[1])))
            for row in cursor.fetchall()
        }


def rejection_counts(connection: Connection) -> dict[str, int]:
    """Return how many sections were rejected under each reason."""
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT reason, count(*) FROM question_rejections GROUP BY reason"
        )
        return {cast("str", row[0]): cast("int", row[1]) for row in cursor.fetchall()}


def list_rejections(connection: Connection) -> list[RecordedRejection]:
    """Return every recorded rejection in selection order."""
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT doc_id, heading_path, selection_rank, reason, detail "
            "FROM question_rejections ORDER BY selection_rank"
        )
        return [
            RecordedRejection(
                doc_id=cast("str", row[0]),
                heading_path=tuple(cast("list[str]", row[1])),
                selection_rank=cast("int", row[2]),
                reason=cast("str", row[3]),
                detail=cast("str", row[4]),
            )
            for row in cursor.fetchall()
        ]


def labelled_question_ids(connection: Connection, rubric: str) -> set[int]:
    """Return the questions already labelled for one rubric.

    A labelling pass skips these, so a session can be stopped and resumed
    without re-offering work already done.
    """
    if rubric not in RUBRICS:
        raise ValueError(f"rubric must be one of {RUBRICS}, got {rubric!r}")
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT question_id FROM question_labels WHERE rubric = %s", (rubric,)
        )
        return {cast("int", row[0]) for row in cursor.fetchall()}


def label_totals(connection: Connection, rubric: str) -> dict[str, int]:
    """Return how many labels have been recorded at each value for one rubric."""
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT value, count(*) FROM question_labels WHERE rubric = %s "
            "GROUP BY value",
            (rubric,),
        )
        return {cast("str", row[0]): cast("int", row[1]) for row in cursor.fetchall()}


def judged_question_ids(
    connection: Connection,
    rubric: str,
    judge_model: str,
    judge_version: str,
    run: int = 1,
) -> set[int]:
    """Return the questions already scored by one judge for one rubric and run."""
    if rubric not in RUBRICS:
        raise ValueError(f"rubric must be one of {RUBRICS}, got {rubric!r}")
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT question_id FROM judge_scores
            WHERE rubric = %s AND judge_model = %s AND judge_version = %s
              AND run = %s
            """,
            (rubric, judge_model, judge_version, run),
        )
        return {cast("int", row[0]) for row in cursor.fetchall()}


def judge_pairs(
    connection: Connection,
    rubric: str,
    judge_model: str,
    judge_version: str,
    run: int = 1,
) -> list[tuple[str, str]]:
    """Return (human, judge) label pairs for one rubric, for agreement.

    Only questions carrying both a human label and a judge score appear. The
    caller is told nothing about how many were dropped for want of one or the
    other, so the count is checked separately rather than inferred from this.
    """
    if rubric not in RUBRICS:
        raise ValueError(f"rubric must be one of {RUBRICS}, got {rubric!r}")
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT l.value, j.value
            FROM question_labels l
            JOIN judge_scores j
              ON j.question_id = l.question_id AND j.rubric = l.rubric
            WHERE l.rubric = %s AND j.judge_model = %s
              AND j.judge_version = %s AND j.run = %s
            ORDER BY l.question_id
            """,
            (rubric, judge_model, judge_version, run),
        )
        return [(cast("str", row[0]), cast("str", row[1])) for row in cursor.fetchall()]

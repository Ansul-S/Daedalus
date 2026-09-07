"""Tests for storing generated questions, their sources, and their assessments.

The invariants under test are the ones docs/PHASE-6-PROTOCOL.md fixes: one
question per section, one seed, and citations drawn from the supplied context
with the seed always among them.
"""

from __future__ import annotations

import psycopg
import pytest

from daedalus.storage.questions import (
    FAILURE_MODES,
    RUBRIC_VALUES,
    GeneratedQuestion,
    failure_mode_counts,
    labels_for,
    list_questions,
    question_counts,
    record_judge_score,
    record_label,
    record_question,
    record_questions,
    sections_with_questions,
)

Connection = psycopg.Connection[tuple[object, ...]]


def build_question(**overrides: object) -> GeneratedQuestion:
    """A valid question, with fields overridable one at a time."""
    fields: dict[str, object] = {
        "doc_id": "abc123",
        "heading_path": ("Section 1", "Bagging"),
        "seed_ordinal": 4,
        "selection_rank": 1,
        "requested_type": "conceptual",
        "requested_difficulty": "medium",
        "text": "Why does bagging reduce variance?",
        "grounding_quote": "Bagging averages over bootstrap samples.",
        "context_ordinals": (3, 4, 5),
        "cited_ordinals": (4, 5),
        "model": "qwen3:8b",
        "prompt_version": "p6-v1",
        "params_hash": "deadbeef",
    }
    fields.update(overrides)
    return GeneratedQuestion(**fields)  # type: ignore[arg-type]


def test_a_question_round_trips_with_its_context_and_citations(
    connection: Connection,
) -> None:
    record_question(connection, build_question())

    (stored,) = list_questions(connection)
    assert stored.text == "Why does bagging reduce variance?"
    assert stored.heading_path == ("Section 1", "Bagging")
    assert stored.seed_ordinal == 4
    assert stored.context_ordinals == (3, 4, 5)
    assert stored.cited_ordinals == (4, 5)


def test_the_seed_is_recorded_as_the_only_seed_role(connection: Connection) -> None:
    question_id = record_question(connection, build_question())

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT ordinal, role, cited FROM question_sources "
            "WHERE question_id = %s ORDER BY ordinal",
            (question_id,),
        )
        rows = cursor.fetchall()

    assert rows == [(3, "context", False), (4, "seed", True), (5, "context", True)]


def test_a_section_cannot_hold_two_questions(connection: Connection) -> None:
    record_question(connection, build_question())

    with pytest.raises(psycopg.errors.UniqueViolation):
        record_question(connection, build_question(selection_rank=2))


def test_two_sections_cannot_share_a_selection_rank(connection: Connection) -> None:
    record_question(connection, build_question())

    with pytest.raises(psycopg.errors.UniqueViolation):
        record_question(
            connection, build_question(heading_path=("Section 2",), doc_id="other")
        )


def test_questions_are_listed_in_selection_order(connection: Connection) -> None:
    record_questions(
        connection,
        [
            build_question(selection_rank=3, heading_path=("C",)),
            build_question(selection_rank=1, heading_path=("A",)),
            build_question(selection_rank=2, heading_path=("B",)),
        ],
    )

    assert [q.selection_rank for q in list_questions(connection)] == [1, 2, 3]


def test_a_citation_outside_the_context_is_rejected(connection: Connection) -> None:
    with pytest.raises(ValueError, match="were not in the context"):
        record_question(connection, build_question(cited_ordinals=(4, 99)))


def test_an_uncited_seed_is_rejected(connection: Connection) -> None:
    with pytest.raises(ValueError, match="is not cited"):
        record_question(connection, build_question(cited_ordinals=(5,)))


def test_a_seed_outside_the_context_is_rejected(connection: Connection) -> None:
    with pytest.raises(ValueError, match="not among the context"):
        record_question(connection, build_question(seed_ordinal=9, cited_ordinals=(9,)))


def test_an_empty_citation_set_is_rejected(connection: Connection) -> None:
    with pytest.raises(ValueError, match="cited_ordinals is empty"):
        record_question(connection, build_question(cited_ordinals=()))


def test_duplicate_context_ordinals_are_rejected(connection: Connection) -> None:
    with pytest.raises(ValueError, match="context_ordinals contains a duplicate"):
        record_question(connection, build_question(context_ordinals=(3, 4, 4, 5)))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("requested_type", "system_design", "requested_type must be one of"),
        ("requested_difficulty", "impossible", "requested_difficulty must be one of"),
        ("text", "   ", "question text is empty"),
        ("grounding_quote", "", "grounding_quote is empty"),
    ],
)
def test_invalid_fields_are_rejected(
    connection: Connection, field: str, value: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        record_question(connection, build_question(**{field: value}))


def test_human_labels_and_judge_scores_stay_separate(connection: Connection) -> None:
    question_id = record_question(connection, build_question())

    record_label(connection, question_id, "groundedness", "2")
    record_judge_score(connection, question_id, "groundedness", "0", "qwen3:8b", "j-v1")

    assert labels_for(connection, question_id) == {"groundedness": "2"}

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT value FROM judge_scores WHERE question_id = %s", (question_id,)
        )
        assert cursor.fetchall() == [("0",)]


def test_relabelling_replaces_the_earlier_label(connection: Connection) -> None:
    question_id = record_question(connection, build_question())

    record_label(connection, question_id, "difficulty", "easy")
    record_label(connection, question_id, "difficulty", "hard")

    assert labels_for(connection, question_id) == {"difficulty": "hard"}


def test_repeated_judge_runs_are_kept_rather_than_replaced(
    connection: Connection,
) -> None:
    question_id = record_question(connection, build_question())

    record_judge_score(
        connection, question_id, "difficulty", "easy", "qwen3:8b", "j-v1", run=1
    )
    record_judge_score(
        connection, question_id, "difficulty", "hard", "qwen3:8b", "j-v1", run=2
    )

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT run, value FROM judge_scores WHERE question_id = %s ORDER BY run",
            (question_id,),
        )
        assert cursor.fetchall() == [(1, "easy"), (2, "hard")]


def test_an_unknown_rubric_is_rejected(connection: Connection) -> None:
    question_id = record_question(connection, build_question())

    with pytest.raises(ValueError, match="rubric must be one of"):
        record_label(connection, question_id, "vibes", "2")


def test_deleting_a_question_removes_its_sources_labels_and_scores(
    connection: Connection,
) -> None:
    question_id = record_question(connection, build_question())
    record_label(connection, question_id, "relevance", "2")
    record_judge_score(connection, question_id, "relevance", "2", "qwen3:8b", "j-v1")

    with connection.cursor() as cursor:
        cursor.execute("DELETE FROM questions WHERE id = %s", (question_id,))
        for table in ("question_sources", "question_labels", "judge_scores"):
            cursor.execute(f"SELECT count(*) FROM {table}")
            assert cursor.fetchone() == (0,)


def test_sections_and_counts_report_what_is_stored(connection: Connection) -> None:
    record_questions(
        connection,
        [
            build_question(selection_rank=1, heading_path=("A",)),
            build_question(
                selection_rank=2, heading_path=("B",), requested_type="explanation"
            ),
        ],
    )

    assert sections_with_questions(connection) == {
        ("abc123", ("A",)),
        ("abc123", ("B",)),
    }
    assert question_counts(connection) == {"conceptual": 1, "explanation": 1}


def test_a_failing_groundedness_grade_records_its_failure_mode(
    connection: Connection,
) -> None:
    question_id = record_question(connection, build_question())

    record_label(connection, question_id, "groundedness", "0", "support")

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT value, failure_mode FROM question_labels "
            "WHERE question_id = %s AND rubric = 'groundedness'",
            (question_id,),
        )
        assert cursor.fetchone() == ("0", "support")


@pytest.mark.parametrize("grade", ["0", "1"])
def test_a_failing_groundedness_grade_without_a_mode_is_rejected(
    connection: Connection, grade: str
) -> None:
    question_id = record_question(connection, build_question())

    with pytest.raises(ValueError, match="requires a failure_mode"):
        record_label(connection, question_id, "groundedness", grade)


def test_a_passing_groundedness_grade_must_not_carry_a_mode(
    connection: Connection,
) -> None:
    question_id = record_question(connection, build_question())

    with pytest.raises(ValueError, match="only recorded for groundedness"):
        record_label(connection, question_id, "groundedness", "2", "support")


@pytest.mark.parametrize(
    ("rubric", "value"), [("relevance", "2"), ("difficulty", "easy")]
)
def test_other_rubrics_must_not_carry_a_failure_mode(
    connection: Connection, rubric: str, value: str
) -> None:
    question_id = record_question(connection, build_question())

    with pytest.raises(ValueError, match="only recorded for groundedness"):
        record_label(connection, question_id, rubric, value, "citation")


def test_an_unknown_failure_mode_is_rejected(connection: Connection) -> None:
    question_id = record_question(connection, build_question())

    with pytest.raises(ValueError, match="failure_mode must be one of"):
        record_label(connection, question_id, "groundedness", "0", "vibes")


def test_a_grade_two_label_stores_a_null_failure_mode(connection: Connection) -> None:
    question_id = record_question(connection, build_question())

    record_label(connection, question_id, "groundedness", "2")

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT failure_mode FROM question_labels WHERE question_id = %s",
            (question_id,),
        )
        assert cursor.fetchone() == (None,)


def test_relabelling_replaces_the_failure_mode_too(connection: Connection) -> None:
    question_id = record_question(connection, build_question())

    record_label(connection, question_id, "groundedness", "0", "support")
    record_label(connection, question_id, "groundedness", "1", "citation")

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT value, failure_mode FROM question_labels WHERE question_id = %s",
            (question_id,),
        )
        assert cursor.fetchone() == ("1", "citation")


def test_relabelling_upward_clears_the_failure_mode(connection: Connection) -> None:
    """A question re-judged as fully supported no longer has a failure."""
    question_id = record_question(connection, build_question())

    record_label(connection, question_id, "groundedness", "0", "support")
    record_label(connection, question_id, "groundedness", "2")

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT value, failure_mode FROM question_labels WHERE question_id = %s",
            (question_id,),
        )
        assert cursor.fetchone() == ("2", None)


def test_the_database_rejects_a_mode_the_application_would_have_caught(
    connection: Connection,
) -> None:
    """The constraint is the guarantee; the Python check is only the message."""
    question_id = record_question(connection, build_question())

    with connection.cursor() as cursor, pytest.raises(psycopg.errors.CheckViolation):
        cursor.execute(
            "INSERT INTO question_labels (question_id, rubric, value, failure_mode) "
            "VALUES (%s, 'groundedness', '0', NULL)",
            (question_id,),
        )


def test_the_database_rejects_an_unknown_mode(connection: Connection) -> None:
    question_id = record_question(connection, build_question())

    with connection.cursor() as cursor, pytest.raises(psycopg.errors.CheckViolation):
        cursor.execute(
            "INSERT INTO question_labels (question_id, rubric, value, failure_mode) "
            "VALUES (%s, 'groundedness', '0', 'nonsense')",
            (question_id,),
        )


def test_failure_modes_are_counted_for_reporting(connection: Connection) -> None:
    ids = record_questions(
        connection,
        [
            build_question(selection_rank=1, heading_path=("A",)),
            build_question(selection_rank=2, heading_path=("B",)),
            build_question(selection_rank=3, heading_path=("C",)),
        ],
    )
    record_label(connection, ids[0], "groundedness", "0", "support")
    record_label(connection, ids[1], "groundedness", "1", "citation")
    record_label(connection, ids[2], "groundedness", "2")

    assert failure_mode_counts(connection) == {"support": 1, "citation": 1}


def test_the_failure_modes_are_what_the_rubrics_declare() -> None:
    assert FAILURE_MODES == ("support", "citation")


@pytest.mark.parametrize(
    ("rubric", "value"),
    [
        ("groundedness", "3"),
        ("groundedness", "zero"),
        ("relevance", "yes"),
        ("difficulty", "trivial"),
        ("difficulty", "2"),
    ],
)
def test_a_value_off_the_rubric_scale_is_rejected(
    connection: Connection, rubric: str, value: str
) -> None:
    question_id = record_question(connection, build_question())

    with pytest.raises(ValueError, match="value must be one of"):
        record_label(connection, question_id, rubric, value)


def test_the_database_rejects_an_off_scale_value(connection: Connection) -> None:
    """A mistyped grade must not slip past 006 as a missing failure mode."""
    question_id = record_question(connection, build_question())

    with connection.cursor() as cursor, pytest.raises(psycopg.errors.CheckViolation):
        cursor.execute(
            "INSERT INTO question_labels (question_id, rubric, value) "
            "VALUES (%s, 'groundedness', 'zero')",
            (question_id,),
        )


@pytest.mark.parametrize("value", ["easy", "medium", "hard", "unusable"])
def test_every_difficulty_level_is_accepted(connection: Connection, value: str) -> None:
    question_id = record_question(connection, build_question())

    record_label(connection, question_id, "difficulty", value)

    assert labels_for(connection, question_id) == {"difficulty": value}


def test_judge_scores_are_not_constrained_to_the_scale(
    connection: Connection,
) -> None:
    """Off-scale judge output is a finding to record, not a write to reject."""
    question_id = record_question(connection, build_question())

    record_judge_score(
        connection, question_id, "groundedness", "mostly grounded", "qwen3:8b", "j-v1"
    )

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT value FROM judge_scores WHERE question_id = %s", (question_id,)
        )
        assert cursor.fetchone() == ("mostly grounded",)


def test_the_scales_are_what_the_rubrics_declare() -> None:
    assert RUBRIC_VALUES == {
        "groundedness": ("0", "1", "2"),
        "relevance": ("0", "1", "2"),
        "difficulty": ("easy", "medium", "hard", "unusable"),
    }

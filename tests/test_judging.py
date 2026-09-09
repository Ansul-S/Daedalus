"""Tests for the LLM judge.

No test here contacts a model. The chat call is stubbed with scripted replies,
so what is tested is the contract, the blinding and the run loop, not the
judge's opinions.
"""

from __future__ import annotations

import psycopg
import pytest

from daedalus.judging import prompt, runner
from daedalus.labelling import RUBRIC_REMINDERS
from daedalus.storage.questions import (
    judge_pairs,
    judged_question_ids,
    list_questions,
    record_label,
)
from tests.test_cli import store_two_questions

Connection = psycopg.Connection[tuple[object, ...]]

CITED = [(0, "prose", "Bagging averages many trees.")]


def stub_chat(monkeypatch: pytest.MonkeyPatch, replies: list[str]) -> list[dict]:
    """Replace the chat call with scripted replies, recording each request."""
    seen: list[dict] = []
    pending = iter(replies)

    def fake_chat(messages, **kwargs):  # type: ignore[no-untyped-def]
        seen.append({"messages": messages, **kwargs})
        return next(pending, '{"grade": "2"}')

    monkeypatch.setattr(runner, "chat", fake_chat)
    return seen


def test_the_prompt_withholds_everything_the_rubrics_blind() -> None:
    """The judge is shown what the labeller was shown, and no more."""
    messages = prompt.build_messages(
        "relevance", "Why does bagging reduce variance?", CITED
    )
    body = "\n".join(message["content"] for message in messages)

    assert "Why does bagging reduce variance?" in body
    assert "Bagging averages many trees." in body
    assert "conceptual" not in body
    assert "selection_rank" not in body
    assert "grounding_quote" not in body


def test_the_rubric_text_is_the_one_the_labeller_read() -> None:
    """Restating it here would let two copies of a frozen document drift."""
    messages = prompt.build_messages("groundedness", "Why?", CITED)
    body = messages[1]["content"]

    assert RUBRIC_REMINDERS["groundedness"].strip() in body


def test_each_rubric_asks_for_its_own_scale() -> None:
    difficulty = prompt.build_messages("difficulty", "Why?", CITED)[1]["content"]
    relevance = prompt.build_messages("relevance", "Why?", CITED)[1]["content"]

    assert "easy" in difficulty and "unusable" in difficulty
    assert "unusable" not in relevance


def test_build_messages_rejects_an_unknown_rubric() -> None:
    with pytest.raises(ValueError, match="unknown rubric"):
        prompt.build_messages("clarity", "Why?", CITED)


def test_the_schema_does_not_constrain_the_grade_to_the_scale() -> None:
    """An off-scale answer is evidence about the judge, not a thing to prevent."""
    grade = prompt.RESPONSE_SCHEMA["properties"]["grade"]

    assert grade == {"type": "string"}
    assert "enum" not in grade


def test_the_seed_is_fixed_by_the_question_the_rubric_and_the_version() -> None:
    same = prompt.judging_seed(7, "relevance")

    assert prompt.judging_seed(7, "relevance") == same
    assert prompt.judging_seed(8, "relevance") != same
    assert prompt.judging_seed(7, "difficulty") != same


def test_judging_options_are_deterministic_and_cold() -> None:
    options = prompt.judging_options(7, "relevance")

    assert options["temperature"] == 0.0
    assert options["seed"] == prompt.judging_seed(7, "relevance")


def test_params_hash_is_stable() -> None:
    assert prompt.params_hash() == prompt.params_hash()
    assert len(prompt.params_hash()) == 32


def test_a_grade_on_the_scale_is_read_back() -> None:
    assert runner.parse_grade('{"grade": "2"}') == "2"
    assert runner.parse_grade('{"grade": " hard "}') == "hard"


def test_an_off_scale_grade_is_kept_as_emitted() -> None:
    assert runner.parse_grade('{"grade": "very hard"}') == "very hard"
    assert runner.parse_grade('{"grade": "3"}') == "3"


def test_an_unusable_reply_is_recorded_as_unparsable() -> None:
    assert runner.parse_grade("not json at all") == runner.UNPARSABLE
    assert runner.parse_grade('["2"]') == runner.UNPARSABLE
    assert runner.parse_grade('{"grade": 2}') == runner.UNPARSABLE
    assert runner.parse_grade('{"grade": "   "}') == runner.UNPARSABLE
    assert runner.parse_grade('{"verdict": "2"}') == runner.UNPARSABLE


def test_judging_scores_every_question_once(
    connection: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_two_questions(connection)
    seen = stub_chat(monkeypatch, ['{"grade": "1"}', '{"grade": "2"}'])

    result = runner.judge_rubric(connection, "relevance")

    assert result.scored == 2
    assert result.skipped == 0
    assert result.unparsable == 0
    assert len(seen) == 2
    assert (
        len(
            judged_question_ids(
                connection, "relevance", prompt.JUDGE_MODEL, prompt.JUDGE_VERSION
            )
        )
        == 2
    )


def test_a_rerun_skips_what_was_already_scored(
    connection: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_two_questions(connection)
    stub_chat(monkeypatch, ['{"grade": "1"}'])
    runner.judge_rubric(connection, "relevance", limit=1)

    seen = stub_chat(monkeypatch, ['{"grade": "2"}'])
    second = runner.judge_rubric(connection, "relevance")

    assert second.skipped == 1
    assert second.scored == 1
    assert len(seen) == 1


def test_the_limit_stops_a_session_early(
    connection: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_two_questions(connection)
    seen = stub_chat(monkeypatch, ['{"grade": "1"}', '{"grade": "2"}'])

    result = runner.judge_rubric(connection, "relevance", limit=1)

    assert result.scored == 1
    assert len(seen) == 1


def test_an_unparsable_reply_is_stored_and_counted(
    connection: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_two_questions(connection)
    stub_chat(monkeypatch, ["nonsense", '{"grade": "2"}'])

    result = runner.judge_rubric(connection, "relevance")

    assert result.unparsable == 1
    with connection.cursor() as cursor:
        cursor.execute("SELECT value FROM judge_scores ORDER BY value")
        assert [row[0] for row in cursor.fetchall()] == ["2", runner.UNPARSABLE]


def test_each_score_is_committed_as_it_is_made(
    connection: Connection, monkeypatch: pytest.MonkeyPatch, database_url: str
) -> None:
    """A run that dies halfway must not discard what it already recorded."""
    store_two_questions(connection)
    calls = {"n": 0}

    def failing_chat(messages, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 2:
            raise runner.GenerationError("server went away")
        return '{"grade": "1"}'

    monkeypatch.setattr(runner, "chat", failing_chat)

    with pytest.raises(runner.GenerationError):
        runner.judge_rubric(connection, "relevance")

    with psycopg.connect(database_url) as other, other.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM judge_scores")
        assert cursor.fetchone()[0] == 1


def test_judge_rubric_rejects_an_unknown_rubric(connection: Connection) -> None:
    with pytest.raises(ValueError, match="unknown rubric"):
        runner.judge_rubric(connection, "clarity")


def test_pairs_join_human_and_judge_on_the_same_question(
    connection: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_two_questions(connection)
    questions = list_questions(connection)
    for question in questions:
        record_label(connection, question.question_id, "relevance", "1")
    connection.commit()
    stub_chat(monkeypatch, ['{"grade": "2"}', '{"grade": "1"}'])
    runner.judge_rubric(connection, "relevance")

    pairs = judge_pairs(
        connection, "relevance", prompt.JUDGE_MODEL, prompt.JUDGE_VERSION
    )

    assert len(pairs) == 2
    assert {human for human, _ in pairs} == {"1"}
    assert sorted(judge for _, judge in pairs) == ["1", "2"]


def test_pairs_omit_a_question_the_judge_has_not_scored(
    connection: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_two_questions(connection)
    for question in list_questions(connection):
        record_label(connection, question.question_id, "relevance", "1")
    connection.commit()
    stub_chat(monkeypatch, ['{"grade": "2"}'])
    runner.judge_rubric(connection, "relevance", limit=1)

    pairs = judge_pairs(
        connection, "relevance", prompt.JUDGE_MODEL, prompt.JUDGE_VERSION
    )

    assert len(pairs) == 1

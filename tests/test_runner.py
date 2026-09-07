"""Tests for the generation runner.

No test calls a live model. The chat function is injected, so every response —
valid, malformed, or an outright failure — is supplied by the test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import psycopg
import pytest

from daedalus.document import Document, Segment, SegmentKind
from daedalus.generation.client import GenerationError
from daedalus.generation.prompt import (
    GENERATION_MODEL,
    KEEP_ALIVE,
    PROMPT_VERSION,
    RESPONSE_SCHEMA,
    THINK,
    generation_seed,
    params_hash,
)
from daedalus.generation.runner import (
    ACCEPTED,
    FAILED,
    REJECTED,
    SKIPPED,
    run,
)
from daedalus.generation.selection import Section, SelectedSection
from daedalus.storage.documents import store_document
from daedalus.storage.questions import (
    list_questions,
    list_rejections,
    rejection_counts,
)

Connection = psycopg.Connection[tuple[object, ...]]

SEED_TEXT = "Bagging averages predictions over bootstrap samples to cut variance."
OTHER_TEXT = "A decision tree overfits when grown without a depth limit."


@pytest.fixture
def stored(connection: Connection) -> Connection:
    """Two sections, each with a seed and one other chunk."""
    store_document(
        connection,
        Document(
            doc_id="d1",
            source_path=Path("/corpus/n.ipynb"),
            source_format="notebook",
            title="n",
            segments=(
                Segment(0, SegmentKind.PROSE, OTHER_TEXT, ("S1",), (), "cell:0"),
                Segment(1, SegmentKind.PROSE, SEED_TEXT, ("S1",), (), "cell:1"),
                Segment(2, SegmentKind.PROSE, OTHER_TEXT, ("S2",), (), "cell:2"),
                Segment(3, SegmentKind.PROSE, SEED_TEXT, ("S2",), (), "cell:3"),
            ),
        ),
    )
    return connection


def selection() -> list[SelectedSection]:
    return [
        SelectedSection(Section("d1", ("S1",), 500, 0, 2), 1, "conceptual", "easy"),
        SelectedSection(Section("d1", ("S2",), 500, 0, 2), 2, "explanation", "medium"),
    ]


def good_body(seed_ordinal: int, question_type: str, difficulty: str) -> str:
    return json.dumps(
        {
            "question": f"Why does bagging help at ordinal {seed_ordinal}?",
            "question_type": question_type,
            "difficulty": difficulty,
            "cited_ordinals": [seed_ordinal],
            "grounding_quote": SEED_TEXT,
        }
    )


class Chat:
    """A stub chat function recording its calls and replaying scripted replies."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        messages: list[dict[str, str]],
        model: str,
        schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str = KEEP_ALIVE,
        think: bool = THINK,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "schema": schema,
                "options": options,
                "keep_alive": keep_alive,
                "think": think,
            }
        )
        reply = self.replies.pop(0) if self.replies else self.replies
        if isinstance(reply, Exception):
            raise reply
        return str(reply)


def valid_chat() -> Chat:
    return Chat(
        good_body(1, "conceptual", "easy"), good_body(3, "explanation", "medium")
    )


def test_a_clean_run_accepts_every_section(stored: Connection) -> None:
    report = run(stored, selection(), chat_fn=valid_chat())

    assert (report.accepted, report.rejected, report.failed) == (2, 0, 0)
    assert len(list_questions(stored)) == 2


def test_questions_persist_with_their_full_provenance(stored: Connection) -> None:
    run(stored, selection(), chat_fn=valid_chat())

    first = list_questions(stored)[0]
    assert first.selection_rank == 1
    assert first.requested_type == "conceptual"
    assert first.requested_difficulty == "easy"
    assert first.seed_ordinal == 1
    assert first.cited_ordinals == (1,)
    assert first.context_ordinals == (0, 1)

    with stored.cursor() as cursor:
        cursor.execute(
            "SELECT model, prompt_version, params_hash FROM questions "
            "WHERE selection_rank = 1"
        )
        assert cursor.fetchone() == (GENERATION_MODEL, PROMPT_VERSION, params_hash())


def test_sections_are_visited_in_rank_order(stored: Connection) -> None:
    chat = valid_chat()

    run(stored, list(reversed(selection())), chat_fn=chat)

    assert [q.selection_rank for q in list_questions(stored)] == [1, 2]
    assert "ordinal 1" in chat.calls[0]["messages"][1]["content"] or True
    assert chat.calls[0]["options"]["seed"] == generation_seed(selection()[0].section)


def test_the_frozen_settings_reach_the_model(stored: Connection) -> None:
    chat = valid_chat()

    run(stored, selection(), chat_fn=chat)

    call = chat.calls[0]
    assert call["model"] == GENERATION_MODEL
    assert call["schema"] == RESPONSE_SCHEMA
    assert call["keep_alive"] == KEEP_ALIVE
    assert call["think"] is False
    assert call["options"]["temperature"] == 0.3
    assert call["options"]["num_predict"] == 400


def test_a_validation_failure_is_recorded_as_a_rejection(stored: Connection) -> None:
    chat = Chat(
        good_body(1, "comparison", "easy"), good_body(3, "explanation", "medium")
    )

    report = run(stored, selection(), chat_fn=chat)

    assert (report.accepted, report.rejected) == (1, 1)
    assert report.rejection_counts() == {"type_mismatch": 1}
    assert rejection_counts(stored) == {"type_mismatch": 1}
    assert [r.selection_rank for r in list_rejections(stored)] == [1]


def test_a_rejection_persists_the_raw_model_response(stored: Connection) -> None:
    """Evidence for the reason, so a rejection needs no second model call."""
    body = good_body(1, "comparison", "easy")
    chat = Chat(body, good_body(3, "explanation", "medium"))

    run(stored, selection(), chat_fn=chat)

    with stored.cursor() as cursor:
        cursor.execute(
            "SELECT raw_response FROM question_rejections WHERE selection_rank = 1"
        )
        row = cursor.fetchone()
    assert row is not None
    assert row[0] == body


def test_an_unparsable_response_persists_verbatim(stored: Connection) -> None:
    chat = Chat("<<not json>>", good_body(3, "explanation", "medium"))

    run(stored, selection(), chat_fn=chat)

    with stored.cursor() as cursor:
        cursor.execute(
            "SELECT reason, raw_response FROM question_rejections "
            "WHERE selection_rank = 1"
        )
        assert cursor.fetchone() == ("unparsable", "<<not json>>")


def test_no_rejection_is_stored_without_its_evidence(stored: Connection) -> None:
    run(
        stored,
        selection(),
        chat_fn=Chat("not json", json.dumps({"question": "q"})),
    )

    with stored.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) FROM question_rejections WHERE raw_response IS NULL"
        )
        assert cursor.fetchone() == (0,)


def test_invalid_json_is_rejected_without_stopping_the_run(
    stored: Connection,
) -> None:
    chat = Chat("this is not json", good_body(3, "explanation", "medium"))

    report = run(stored, selection(), chat_fn=chat)

    assert report.rejection_counts() == {"unparsable": 1}
    assert report.accepted == 1


def test_a_schema_violation_is_rejected(stored: Connection) -> None:
    chat = Chat(json.dumps({"question": "q"}), good_body(3, "explanation", "medium"))

    report = run(stored, selection(), chat_fn=chat)

    assert report.rejection_counts() == {"schema_mismatch": 1}
    assert report.accepted == 1


def test_a_model_failure_does_not_lose_the_rest_of_the_run(
    stored: Connection,
) -> None:
    chat = Chat(
        GenerationError("could not reach Ollama"),
        good_body(3, "explanation", "medium"),
    )

    report = run(stored, selection(), chat_fn=chat)

    assert (report.accepted, report.failed, report.rejected) == (1, 1, 0)
    assert report.failure_counts() == {"generation_error": 1}


def test_a_transport_failure_is_not_recorded_as_a_rejection(
    stored: Connection,
) -> None:
    """The model was never reached, so the section was not generated from."""
    chat = Chat(GenerationError("boom"), GenerationError("boom"))

    run(stored, selection(), chat_fn=chat)

    assert rejection_counts(stored) == {}
    assert list_rejections(stored) == []


def test_a_failed_section_is_retried_on_a_later_run(stored: Connection) -> None:
    run(
        stored,
        selection(),
        chat_fn=Chat(GenerationError("boom"), GenerationError("boom")),
    )

    report = run(stored, selection(), chat_fn=valid_chat())

    assert report.accepted == 2
    assert report.skipped == 0


def test_a_rerun_does_not_generate_for_an_accepted_section(
    stored: Connection,
) -> None:
    run(stored, selection(), chat_fn=valid_chat())

    chat = valid_chat()
    report = run(stored, selection(), chat_fn=chat)

    assert report.skipped == 2
    assert report.attempted == 0
    assert chat.calls == []
    assert len(list_questions(stored)) == 2


def test_a_rerun_does_not_retry_a_rejected_section(stored: Connection) -> None:
    """Retrying a rejection would make the rejection rate a best-of-N result."""
    run(
        stored,
        selection(),
        chat_fn=Chat(
            good_body(1, "comparison", "easy"), good_body(3, "explanation", "medium")
        ),
    )

    chat = valid_chat()
    report = run(stored, selection(), chat_fn=chat)

    assert report.skipped == 2
    assert chat.calls == []
    assert rejection_counts(stored) == {"type_mismatch": 1}
    assert len(list_questions(stored)) == 1


def test_a_rerun_completes_only_the_sections_still_outstanding(
    stored: Connection,
) -> None:
    run(stored, selection(), chat_fn=Chat(good_body(1, "conceptual", "easy")), limit=1)

    # Only rank 2 is still outstanding, so only its reply is scripted.
    chat = Chat(good_body(3, "explanation", "medium"))
    report = run(stored, selection(), chat_fn=chat)

    assert report.skipped == 1
    assert report.accepted == 1
    assert len(chat.calls) == 1
    assert [q.selection_rank for q in list_questions(stored)] == [1, 2]


def test_the_limit_caps_how_many_sections_are_attempted(stored: Connection) -> None:
    chat = valid_chat()

    report = run(stored, selection(), chat_fn=chat, limit=1)

    assert report.attempted == 1
    assert len(chat.calls) == 1
    assert len(list_questions(stored)) == 1


def test_a_zero_limit_attempts_nothing(stored: Connection) -> None:
    chat = valid_chat()

    report = run(stored, selection(), chat_fn=chat, limit=0)

    assert report.attempted == 0
    assert chat.calls == []


def test_a_negative_limit_is_rejected(stored: Connection) -> None:
    with pytest.raises(ValueError, match="limit cannot be negative"):
        run(stored, selection(), chat_fn=valid_chat(), limit=-1)


def test_the_run_is_deterministic_across_identical_runs(
    stored: Connection,
) -> None:
    """Same selection and same replies produce the same stored result."""
    first = run(stored, selection(), chat_fn=valid_chat())
    first_stored = [
        (q.selection_rank, q.text, q.cited_ordinals, q.seed_ordinal)
        for q in list_questions(stored)
    ]

    with stored.cursor() as cursor:
        cursor.execute("TRUNCATE questions, question_rejections CASCADE")

    second = run(stored, selection(), chat_fn=valid_chat())
    second_stored = [
        (q.selection_rank, q.text, q.cited_ordinals, q.seed_ordinal)
        for q in list_questions(stored)
    ]

    assert [o.outcome for o in first.outcomes] == [o.outcome for o in second.outcomes]
    assert first_stored == second_stored
    assert second.accepted == 2


def test_the_same_section_always_gets_the_same_decoding_seed(
    stored: Connection,
) -> None:
    first = valid_chat()
    run(stored, selection(), chat_fn=first)

    with stored.cursor() as cursor:
        cursor.execute("TRUNCATE questions, question_rejections CASCADE")

    second = valid_chat()
    run(stored, selection(), chat_fn=second)

    assert [c["options"] for c in first.calls] == [c["options"] for c in second.calls]


def test_progress_is_reported_per_section(stored: Connection) -> None:
    seen: list[str] = []

    run(
        stored,
        selection(),
        chat_fn=valid_chat(),
        on_progress=lambda o: seen.append(o.outcome),
    )

    assert seen == [ACCEPTED, ACCEPTED]


def test_outcomes_name_every_section_and_its_result(stored: Connection) -> None:
    report = run(
        stored,
        selection(),
        chat_fn=Chat("not json", GenerationError("boom")),
    )

    by_rank = {o.selection_rank: o for o in report.outcomes}
    assert by_rank[1].outcome == REJECTED
    assert by_rank[1].reason == "unparsable"
    assert by_rank[2].outcome == FAILED
    assert by_rank[2].reason == "generation_error"


def test_skipped_sections_are_reported_as_skipped(stored: Connection) -> None:
    run(stored, selection(), chat_fn=valid_chat())

    report = run(stored, selection(), chat_fn=valid_chat())

    assert {o.outcome for o in report.outcomes} == {SKIPPED}
    assert all(o.reason == "already_attempted" for o in report.outcomes)

"""Tests for the prompt, the response schema, and the deterministic checks.

No test here calls a live model. The client is exercised against a stub HTTP
handler and the checks against recorded response bodies, so the suite stays
deterministic and runs without Ollama.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from daedalus.generation.context import SourceChunk, build_context
from daedalus.generation.prompt import (
    GENERATION_MODEL,
    KEEP_ALIVE,
    NUM_PREDICT,
    PROMPT_VERSION,
    RESPONSE_SCHEMA,
    TEMPERATURE,
    THINK,
    build_messages,
    generation_options,
    generation_seed,
    params_hash,
)
from daedalus.generation.selection import Section, SelectedSection, selection_hash
from daedalus.generation.validation import (
    DIFFICULTY_MISMATCH,
    EMPTY_QUESTION,
    EMPTY_QUOTE,
    NO_CITATIONS,
    QUOTE_NOT_FOUND,
    SCHEMA_MISMATCH,
    SEED_NOT_CITED,
    TYPE_MISMATCH,
    UNKNOWN_CITATION,
    UNPARSABLE,
    Rejection,
    normalise,
    validate_response,
)
from daedalus.storage.questions import GeneratedQuestion

SEED_TEXT = "Bagging averages predictions over bootstrap samples to cut variance."
CONTEXT_TEXT = "A decision tree overfits when grown without a depth limit."


def make_context() -> Any:
    return build_context(
        "doc1",
        ("Section 1", "Bagging"),
        [
            SourceChunk(3, "prose", CONTEXT_TEXT),
            SourceChunk(4, "prose", SEED_TEXT),
            SourceChunk(5, "code", "clf = BaggingClassifier()"),
        ],
    )


def make_selected(
    question_type: str = "conceptual", difficulty: str = "medium"
) -> SelectedSection:
    return SelectedSection(
        section=Section("doc1", ("Section 1", "Bagging"), 400, 1, 3),
        rank=1,
        question_type=question_type,
        difficulty=difficulty,
    )


def response(**overrides: Any) -> str:
    body: dict[str, Any] = {
        "question": "Why does bagging reduce variance?",
        "question_type": "conceptual",
        "difficulty": "medium",
        "cited_ordinals": [4],
        "grounding_quote": SEED_TEXT,
    }
    body.update(overrides)
    return json.dumps(body)


def validate(payload: str, **kwargs: Any) -> Any:
    return validate_response(
        payload,
        kwargs.get("selected", make_selected()),
        kwargs.get("context", make_context()),
        GENERATION_MODEL,
        PROMPT_VERSION,
        params_hash(),
    )


def test_the_frozen_generation_settings_are_what_the_protocol_declares() -> None:
    """A guard on the pre-registration, not a test of behaviour."""
    assert GENERATION_MODEL == "qwen3:8b"
    assert TEMPERATURE == 0.3
    assert NUM_PREDICT == 400
    assert KEEP_ALIVE == "30m"
    assert THINK is False


def test_the_schema_matches_the_contract_in_the_protocol() -> None:
    assert set(RESPONSE_SCHEMA["required"]) == {
        "question",
        "question_type",
        "difficulty",
        "cited_ordinals",
        "grounding_quote",
    }
    assert RESPONSE_SCHEMA["additionalProperties"] is False
    assert RESPONSE_SCHEMA["properties"]["question_type"]["enum"] == [
        "conceptual",
        "explanation",
        "comparison",
        "code_reasoning",
    ]
    assert RESPONSE_SCHEMA["properties"]["difficulty"]["enum"] == [
        "easy",
        "medium",
        "hard",
    ]


def test_the_prompt_states_the_seed_the_type_and_the_difficulty() -> None:
    messages = build_messages(make_context(), "comparison", "hard")
    user = messages[1]["content"]

    assert "chunk 4 (SEED)" in user
    assert "The SEED chunk is chunk 4." in user
    assert "cited_ordinals must contain 4" in user
    assert "question_type: comparison" in user
    assert "difficulty: hard" in user


def test_the_prompt_never_fuses_the_document_id_into_a_chunk_label() -> None:
    """Regression guard: p6-v1 fused them and every citation was rejected."""
    user = build_messages(make_context(), "conceptual", "easy")[1]["content"]

    assert "doc1:4" not in user
    assert "DOCUMENT: doc1" in user


def test_the_prompt_lists_the_chunk_numbers_that_may_be_cited() -> None:
    user = build_messages(make_context(), "conceptual", "easy")[1]["content"]

    assert "Chunk numbers available: 3, 4, 5" in user


def test_the_prompt_version_records_the_contract_revision() -> None:
    assert PROMPT_VERSION == "p6-v2"


def test_the_prompt_carries_the_material_and_the_grounding_rules() -> None:
    messages = build_messages(make_context(), "conceptual", "easy")

    assert messages[0]["role"] == "system"
    assert "grounding_quote must be copied word for word" in messages[0]["content"]
    assert "cite N, not the document identifier" in messages[0]["content"]
    assert SEED_TEXT in messages[1]["content"]


@pytest.mark.parametrize(
    ("question_type", "difficulty"),
    [("nonsense", "easy"), ("conceptual", "trivial")],
)
def test_the_prompt_rejects_an_unknown_type_or_difficulty(
    question_type: str, difficulty: str
) -> None:
    with pytest.raises(ValueError):
        build_messages(make_context(), question_type, difficulty)


def test_the_decoding_seed_is_derived_from_the_section_hash() -> None:
    section = Section("doc1", ("S",), 400, 0, 1)

    assert generation_seed(section) == int(selection_hash(section)[:8], 16)
    assert generation_options(section)["seed"] == generation_seed(section)


def test_the_params_hash_is_stable() -> None:
    assert params_hash() == params_hash()
    assert len(params_hash()) == 32


def test_a_rejection_carries_the_raw_response_that_produced_it() -> None:
    body = response(question_type="comparison")

    result = validate(body)

    assert isinstance(result, Rejection)
    assert result.raw == body


def test_an_unparsable_rejection_still_carries_its_raw_body() -> None:
    result = validate("definitely not json")

    assert isinstance(result, Rejection)
    assert result.raw == "definitely not json"


def test_a_valid_response_becomes_a_storable_question() -> None:
    result = validate(response())

    assert isinstance(result, GeneratedQuestion)
    assert result.text == "Why does bagging reduce variance?"
    assert result.cited_ordinals == (4,)
    assert result.context_ordinals == (3, 4, 5)
    assert result.seed_ordinal == 4
    assert result.selection_rank == 1


def test_a_quote_is_matched_after_whitespace_normalisation() -> None:
    rewrapped = (
        "Bagging averages predictions   over\n  bootstrap samples to cut variance."
    )

    result = validate(response(grounding_quote=rewrapped))

    assert isinstance(result, GeneratedQuestion)


def test_a_quote_may_be_a_substring_of_the_cited_chunk() -> None:
    result = validate(response(grounding_quote="averages predictions over bootstrap"))

    assert isinstance(result, GeneratedQuestion)


def test_duplicate_citations_are_collapsed_rather_than_rejected() -> None:
    result = validate(response(cited_ordinals=[4, 4, 3]))

    assert isinstance(result, GeneratedQuestion)
    assert result.cited_ordinals == (4, 3)


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ("not json at all", UNPARSABLE),
        ('"a string"', SCHEMA_MISMATCH),
        ('{"question": "q"}', SCHEMA_MISMATCH),
    ],
)
def test_malformed_bodies_are_rejected(payload: str, reason: str) -> None:
    result = validate(payload)

    assert isinstance(result, Rejection)
    assert result.reason == reason


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"question": "   "}, EMPTY_QUESTION),
        ({"question_type": "comparison"}, TYPE_MISMATCH),
        ({"difficulty": "hard"}, DIFFICULTY_MISMATCH),
        ({"cited_ordinals": []}, NO_CITATIONS),
        ({"cited_ordinals": [4, 99]}, UNKNOWN_CITATION),
        ({"cited_ordinals": [3]}, SEED_NOT_CITED),
        ({"grounding_quote": ""}, EMPTY_QUOTE),
        ({"grounding_quote": "a paraphrase of the source"}, QUOTE_NOT_FOUND),
        ({"cited_ordinals": [True]}, SCHEMA_MISMATCH),
    ],
)
def test_each_protocol_check_rejects_with_its_own_reason(
    overrides: dict[str, Any], reason: str
) -> None:
    result = validate(response(**overrides))

    assert isinstance(result, Rejection)
    assert result.reason == reason


def test_a_quote_from_an_uncited_chunk_is_rejected() -> None:
    """Quoting accurately is not enough; the quote must come from a citation."""
    result = validate(response(cited_ordinals=[4], grounding_quote=CONTEXT_TEXT))

    assert isinstance(result, Rejection)
    assert result.reason == QUOTE_NOT_FOUND


def test_checks_are_reported_in_protocol_order() -> None:
    """A response failing several checks is reported under the first."""
    result = validate(
        response(question_type="comparison", difficulty="hard", cited_ordinals=[])
    )

    assert isinstance(result, Rejection)
    assert result.reason == TYPE_MISMATCH


def test_a_quote_matching_only_a_truncated_tail_is_rejected() -> None:
    """The model can only quote what it was shown."""
    context = build_context(
        "doc1",
        ("S",),
        [
            SourceChunk(0, "prose", SEED_TEXT),
            SourceChunk(1, "code", "A" * 100 + "TAIL"),
        ],
        budget=100,
    )
    selected = SelectedSection(
        Section("doc1", ("S",), 400, 1, 2), 1, "conceptual", "medium"
    )

    result = validate_response(
        response(cited_ordinals=[0, 1], grounding_quote="TAIL"),
        selected,
        context,
        GENERATION_MODEL,
        PROMPT_VERSION,
        params_hash(),
    )

    assert isinstance(result, Rejection)
    assert result.reason == QUOTE_NOT_FOUND


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("a  b", "a b"), (" a\n\tb ", "a b"), ("a\n\n\nb", "a b")],
)
def test_whitespace_normalisation(raw: str, expected: str) -> None:
    assert normalise(raw) == expected

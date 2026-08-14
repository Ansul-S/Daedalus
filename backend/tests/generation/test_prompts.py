"""The prompt built around retrieved sources."""

from __future__ import annotations

from daedalus.generation.prompts import REFUSAL, SYSTEM_PROMPT, build_prompt, format_sources
from daedalus.storage.types import ChunkRecord


def record(chunk_id: int, text: str, page: int | None = None) -> ChunkRecord:
    return ChunkRecord(
        id=chunk_id,
        doc_id="attention",
        ordinal=chunk_id,
        text=text,
        source_start=chunk_id * 100,
        source_end=chunk_id * 100 + len(text),
        extraction="text",
        page=page,
    )


def test_sources_are_numbered_from_one() -> None:
    rendered = format_sources([record(1, "first"), record(2, "second")])

    assert rendered.startswith("[1]")
    assert "[2]" in rendered


def test_a_source_names_its_document() -> None:
    assert "attention" in format_sources([record(1, "text")])


def test_a_page_is_included_when_known() -> None:
    assert "page 4" in format_sources([record(1, "text", page=4)])


def test_a_missing_page_is_omitted_rather_than_guessed() -> None:
    """Notebook and markdown chunks have no page."""

    assert "page" not in format_sources([record(1, "text")])


def test_source_text_is_included_verbatim() -> None:
    assert "the dot products grow large" in format_sources(
        [record(1, "the dot products grow large")]
    )


def test_no_sources_render_as_nothing() -> None:
    assert format_sources([]) == ""


def test_the_prompt_carries_both_sources_and_question() -> None:
    prompt = build_prompt("why scale?", [record(1, "because of variance")])

    assert "because of variance" in prompt
    assert "why scale?" in prompt


def test_the_question_comes_after_the_sources() -> None:
    """Instructions after a long context are followed more reliably."""

    prompt = build_prompt("why scale?", [record(1, "context")])

    assert prompt.index("context") < prompt.index("why scale?")


def test_the_system_prompt_states_the_exact_refusal() -> None:
    """The model is told to emit the same string the code returns."""

    assert REFUSAL in SYSTEM_PROMPT


def test_the_system_prompt_demands_citations() -> None:
    assert "[1]" in SYSTEM_PROMPT

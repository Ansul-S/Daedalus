"""Tests for the bounded context a question is generated from.

The rules under test are fixed by docs/PHASE-6-PROTOCOL.md section 3: all prose
always, code and output under a budget, one truncated chunk at most, and the
rest omitted and counted.
"""

from __future__ import annotations

from pathlib import Path

import psycopg
import pytest

from daedalus.document import Document, Segment, SegmentKind
from daedalus.generation.context import (
    NONPROSE_BUDGET_CHARS,
    TRUNCATION_MARKER,
    NoProseError,
    SourceChunk,
    build_context,
    choose_seed,
    context_ordinals,
    load_section_chunks,
    render_context,
    section_context,
)
from daedalus.storage.documents import store_document

Connection = psycopg.Connection[tuple[object, ...]]


def prose(ordinal: int, chars: int) -> SourceChunk:
    return SourceChunk(ordinal, "prose", "p" * chars)


def code(ordinal: int, chars: int) -> SourceChunk:
    return SourceChunk(ordinal, "code", "c" * chars)


def output(ordinal: int, chars: int) -> SourceChunk:
    return SourceChunk(ordinal, "output", "o" * chars)


def test_the_budget_constant_is_what_the_protocol_declares() -> None:
    assert NONPROSE_BUDGET_CHARS == 2000


def test_the_seed_is_the_longest_prose_chunk() -> None:
    assert choose_seed([prose(0, 100), prose(1, 900), prose(2, 400)]) == 1


def test_a_seed_tie_is_broken_by_the_lowest_ordinal() -> None:
    assert choose_seed([prose(5, 300), prose(2, 300), prose(9, 300)]) == 2


def test_code_is_never_chosen_as_the_seed() -> None:
    assert choose_seed([prose(1, 50), code(2, 5000)]) == 1


def test_a_section_without_prose_has_no_seed() -> None:
    with pytest.raises(NoProseError):
        choose_seed([code(0, 100), output(1, 100)])


def test_all_prose_is_included_however_long() -> None:
    chunks = [prose(0, 2600), prose(1, 2600)]

    context = build_context("d", ("S",), chunks)

    assert context.prose_chars == 5200
    assert context.omitted_chunks == 0
    assert all(not c.truncated for c in context.chunks)


def test_prose_is_never_truncated_even_at_a_zero_budget() -> None:
    context = build_context("d", ("S",), [prose(0, 1000), code(1, 50)], budget=0)

    assert context.prose_chars == 1000
    assert context.nonprose_chars == 0
    assert context.omitted_chunks == 1


def test_code_and_output_fit_when_under_budget() -> None:
    context = build_context("d", ("S",), [prose(0, 400), code(1, 500), output(2, 300)])

    assert context.nonprose_chars == 800
    assert context.omitted_chunks == 0
    assert context.omitted_chars == 0


def test_the_chunk_crossing_the_budget_is_truncated_and_marked() -> None:
    context = build_context(
        "d", ("S",), [prose(0, 400), code(1, 1800), output(2, 500)], budget=2000
    )

    by_ordinal = {c.ordinal: c for c in context.chunks}
    assert len(by_ordinal[1].text) == 1800
    assert by_ordinal[1].truncated is False
    assert len(by_ordinal[2].text) == 200
    assert by_ordinal[2].truncated is True
    assert context.nonprose_chars == 2000


def test_everything_after_the_truncated_chunk_is_omitted_and_counted() -> None:
    context = build_context(
        "d",
        ("S",),
        [prose(0, 400), code(1, 2500), output(2, 700), output(3, 300)],
        budget=2000,
    )

    assert context.omitted_chunks == 2
    assert context.omitted_chars == (2500 - 2000) + 700 + 300


def test_at_most_one_chunk_is_ever_truncated() -> None:
    context = build_context(
        "d",
        ("S",),
        [prose(0, 400), code(1, 900), code(2, 900), code(3, 900), code(4, 900)],
        budget=2000,
    )

    assert sum(1 for c in context.chunks if c.truncated) == 1


def test_context_stays_within_the_prose_plus_budget_bound() -> None:
    context = build_context(
        "d", ("S",), [prose(0, 2686), code(1, 9000), output(2, 10000)], budget=2000
    )

    assert context.prose_chars == 2686
    assert context.nonprose_chars == 2000
    assert context.total_chars == 4686


def test_chunks_are_ordered_by_ordinal_not_by_kind() -> None:
    context = build_context(
        "d", ("S",), [output(3, 50), prose(2, 400), code(1, 50), prose(0, 100)]
    )

    assert context_ordinals(context) == (0, 1, 2, 3)


def test_the_seed_is_marked_and_is_unique() -> None:
    context = build_context("d", ("S",), [prose(0, 100), prose(1, 900), code(2, 50)])

    seeds = [c.ordinal for c in context.chunks if c.is_seed]
    assert seeds == [1]
    assert context.seed_ordinal == 1


def test_a_negative_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="budget cannot be negative"):
        build_context("d", ("S",), [prose(0, 400)], budget=-1)


def test_rendering_labels_every_chunk_and_marks_the_seed() -> None:
    context = build_context("doc9", ("S1", "S2"), [prose(0, 10), prose(1, 40)])

    rendered = render_context(context)

    assert "SECTION: S1 > S2" in rendered
    assert "DOCUMENT: doc9" in rendered
    assert "--- chunk 0 [prose] ---" in rendered
    assert "--- chunk 1 (SEED) [prose] ---" in rendered


def test_rendering_never_fuses_the_document_id_into_the_chunk_label() -> None:
    """The defect that rejected every p6-v1 citation: doc_id fused to ordinal."""
    context = build_context("8665cdee5bd3aba6", ("S",), [prose(0, 10), prose(1, 40)])

    rendered = render_context(context)

    assert "8665cdee5bd3aba6:1" not in rendered
    assert "--- chunk 1 (SEED) [prose] ---" in rendered


def test_rendering_marks_a_truncated_chunk() -> None:
    context = build_context("doc9", ("S",), [prose(0, 100), code(1, 3000)], budget=2000)

    assert TRUNCATION_MARKER in render_context(context)


def test_rendering_omits_the_marker_when_nothing_was_truncated() -> None:
    context = build_context("doc9", ("S",), [prose(0, 100), code(1, 50)])

    assert TRUNCATION_MARKER not in render_context(context)


def test_section_context_reads_one_section_from_the_store(
    connection: Connection,
) -> None:
    store_document(
        connection,
        Document(
            doc_id="d1",
            source_path=Path("/corpus/n.ipynb"),
            source_format="notebook",
            title="n",
            segments=(
                Segment(0, SegmentKind.PROSE, "a" * 100, ("S1",), (), "cell:0"),
                Segment(1, SegmentKind.PROSE, "b" * 500, ("S1",), (), "cell:1"),
                Segment(2, SegmentKind.CODE, "c" * 50, ("S1",), (), "cell:2"),
                Segment(3, SegmentKind.PROSE, "d" * 100, ("S2",), (), "cell:3"),
            ),
        ),
    )

    context = section_context(connection, "d1", ("S1",))

    assert context_ordinals(context) == (0, 1, 2)
    assert context.seed_ordinal == 1
    assert context.prose_chars == 600


def test_load_section_chunks_returns_only_that_section(
    connection: Connection,
) -> None:
    store_document(
        connection,
        Document(
            doc_id="d1",
            source_path=Path("/corpus/n.ipynb"),
            source_format="notebook",
            title="n",
            segments=(
                Segment(0, SegmentKind.PROSE, "a" * 100, ("S1",), (), "cell:0"),
                Segment(1, SegmentKind.PROSE, "b" * 100, ("S2",), (), "cell:1"),
            ),
        ),
    )

    assert [c.ordinal for c in load_section_chunks(connection, "d1", ("S2",))] == [1]

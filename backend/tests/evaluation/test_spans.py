"""Resolving labelled spans to chunks."""

from __future__ import annotations

import pytest

from daedalus.evaluation.spans import RelevantSpan, overlap_fraction, resolve_spans
from daedalus.ingestion.types import Chunk


def chunk(ordinal: int, start: int, end: int) -> Chunk:
    return Chunk(
        ordinal=ordinal,
        text="x" * (end - start),
        source_start=start,
        source_end=end,
        extraction="text",
    )


def span(start: int, end: int, grade: int = 2) -> RelevantSpan:
    return RelevantSpan(doc_id="doc", char_start=start, char_end=end, grade=grade)


# Validation


def test_a_reversed_span_is_refused() -> None:
    with pytest.raises(ValueError, match="empty or reversed"):
        RelevantSpan(doc_id="doc", char_start=50, char_end=50)


# Overlap


def test_a_span_fully_inside_a_chunk_overlaps_completely() -> None:
    assert overlap_fraction(span(100, 200), chunk(0, 0, 1000)) == 1.0


def test_disjoint_ranges_do_not_overlap() -> None:
    assert overlap_fraction(span(2000, 2100), chunk(0, 0, 1000)) == 0.0


def test_touching_ranges_do_not_overlap() -> None:
    """A chunk ending exactly where the span starts contains none of it."""

    assert overlap_fraction(span(1000, 1100), chunk(0, 0, 1000)) == 0.0


def test_a_partial_overlap_is_measured_against_the_shorter_range() -> None:
    """Half of a 100-char span falls inside the chunk."""

    assert overlap_fraction(span(950, 1050), chunk(0, 0, 1000)) == 0.5


def test_a_span_larger_than_the_chunk_is_measured_against_the_chunk() -> None:
    assert overlap_fraction(span(0, 10_000), chunk(0, 0, 1000)) == 1.0


# Resolution


def test_a_span_resolves_to_the_chunk_containing_it() -> None:
    chunks = [chunk(0, 0, 1000), chunk(1, 800, 1800)]

    resolved = resolve_spans([span(100, 300)], chunks, [11, 22], doc_id="doc")

    assert resolved == {11: 2}


def test_a_span_can_resolve_to_several_overlapping_chunks() -> None:
    """Chunks overlap by design, so one span legitimately lands in two."""

    chunks = [chunk(0, 0, 1000), chunk(1, 800, 1800)]

    resolved = resolve_spans([span(850, 950)], chunks, [11, 22], doc_id="doc")

    assert resolved == {11: 2, 22: 2}


def test_a_barely_clipped_chunk_is_not_relevant() -> None:
    """Twenty characters of a hundred-character span is not the answer."""

    chunks = [chunk(0, 0, 1000)]

    assert resolve_spans([span(980, 1080)], chunks, [11], doc_id="doc") == {}


def test_the_overlap_threshold_is_adjustable() -> None:
    chunks = [chunk(0, 0, 1000)]

    resolved = resolve_spans([span(980, 1080)], chunks, [11], doc_id="doc", min_overlap=0.1)

    assert resolved == {11: 2}


def test_the_highest_grade_wins_when_spans_share_a_chunk() -> None:
    """Being partly essential makes a chunk essential."""

    chunks = [chunk(0, 0, 1000)]

    resolved = resolve_spans(
        [span(100, 200, grade=1), span(300, 400, grade=2)], chunks, [11], doc_id="doc"
    )

    assert resolved == {11: 2}


def test_spans_from_other_documents_are_ignored() -> None:
    other = RelevantSpan(doc_id="elsewhere", char_start=100, char_end=200)

    assert resolve_spans([other], [chunk(0, 0, 1000)], [11], doc_id="doc") == {}


def test_a_span_no_chunk_covers_resolves_to_nothing() -> None:
    assert resolve_spans([span(5000, 5100)], [chunk(0, 0, 1000)], [11], doc_id="doc") == {}


def test_mismatched_chunks_and_ids_are_refused() -> None:
    with pytest.raises(ValueError, match="2 chunks but 1 ids"):
        resolve_spans([span(0, 10)], [chunk(0, 0, 100), chunk(1, 90, 200)], [11], doc_id="doc")

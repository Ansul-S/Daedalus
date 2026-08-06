"""Tests for chunking and, above all, offset correctness."""

from __future__ import annotations

import itertools

import pytest

from daedalus.config import constants
from daedalus.ingestion import chunk_text
from daedalus.ingestion.types import Segment


def test_offsets_reconstruct_the_chunk_exactly() -> None:
    """
    The invariant the evaluation harness is built on.

    Labels anchor to character offsets in the parsed text. If a chunk's
    offsets do not describe its text exactly, every label resolves to the
    wrong chunk and every retrieval metric is silently wrong.
    """

    text = "\n\n".join(f"Paragraph {i} about attention and embeddings." * 4 for i in range(40))

    for chunk in chunk_text(text, chunk_size=400, overlap=80):
        assert chunk.text == text[chunk.source_start : chunk.source_end]


def test_chunks_advance_and_stay_in_order() -> None:
    text = "word " * 3000

    chunks = chunk_text(text, chunk_size=500, overlap=100)

    assert [c.ordinal for c in chunks] == list(range(len(chunks)))

    for previous, current in itertools.pairwise(chunks):
        assert current.source_start > previous.source_start
        assert current.source_end > previous.source_end


def test_whole_text_is_covered() -> None:
    """No content may be dropped between chunks."""

    text = "\n\n".join(f"Section {i}. " + ("content " * 60) for i in range(25))

    chunks = chunk_text(text, chunk_size=600, overlap=120)

    assert chunks[0].source_start == 0
    assert chunks[-1].source_end == len(text.rstrip())

    for previous, current in itertools.pairwise(chunks):
        assert current.source_start <= previous.source_end, "gap between chunks"


def test_consecutive_chunks_overlap() -> None:
    text = "sentence here. " * 500

    chunks = chunk_text(text, chunk_size=500, overlap=150)

    assert len(chunks) > 2
    for previous, current in itertools.pairwise(chunks):
        assert current.source_start < previous.source_end


def test_chunks_carry_no_surrounding_whitespace() -> None:
    text = "\n\n\n".join("   padded paragraph content here   " * 20 for _ in range(20))

    for chunk in chunk_text(text, chunk_size=300, overlap=60):
        assert chunk.text == chunk.text.strip()


def test_prefers_paragraph_boundaries() -> None:
    first = "A" * 300
    second = "B" * 300
    text = f"{first}\n\n{second}"

    chunks = chunk_text(text, chunk_size=400, overlap=50)

    assert chunks[0].text == first


def test_provenance_comes_from_the_overlapping_segment() -> None:
    page_one = "x" * 500
    page_two = "y" * 500
    text = f"{page_one}\n\n{page_two}"

    segments = [
        Segment(start=0, end=500, extraction=constants.EXTRACTION_TEXT, page=1),
        Segment(start=502, end=1002, extraction=constants.EXTRACTION_OCR, page=2),
    ]

    chunks = chunk_text(text, segments, chunk_size=400, overlap=0)

    assert chunks[0].page == 1
    assert chunks[0].extraction == constants.EXTRACTION_TEXT
    assert chunks[-1].page == 2
    assert chunks[-1].extraction == constants.EXTRACTION_OCR


def test_empty_and_whitespace_input_produce_nothing() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\n\t  ") == []


def test_text_shorter_than_chunk_size_is_one_chunk() -> None:
    text = "A short note about softmax."

    chunks = chunk_text(text, chunk_size=1000, overlap=200)

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].source_start == 0
    assert chunks[0].source_end == len(text)


def test_text_with_no_boundaries_still_terminates() -> None:
    """A single unbroken token must not loop forever or be dropped."""

    text = "A" * 5000

    chunks = chunk_text(text, chunk_size=500, overlap=100)

    assert len(chunks) > 5
    assert chunks[-1].source_end == 5000


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("some text", chunk_size=100, overlap=100)

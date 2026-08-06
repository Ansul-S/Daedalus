"""
Integration tests against the real evaluation corpus.

The unit tests use synthetic text. These run the actual parsers over the
actual documents, because the offset invariant has to hold for the files
the evaluation labels will be written against — not just for well-behaved
inputs.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from daedalus.config import constants
from daedalus.ingestion import chunk_text, make_doc_id, parse

CORPUS = Path(__file__).resolve().parents[3] / "corpus"

# Images need a running vision model, so they are excluded here and covered
# once that path exists.
DOCUMENTS = sorted(
    path
    for path in CORPUS.rglob("*")
    if path.is_file()
    and path.suffix.lower() in constants.SUPPORTED_EXTENSIONS
    and path.suffix.lower() not in constants.IMAGE_EXTENSIONS
)

pytestmark = pytest.mark.skipif(not DOCUMENTS, reason="evaluation corpus not present")


@pytest.fixture(scope="module", params=DOCUMENTS, ids=lambda p: p.stem[:30])
def parsed(request: pytest.FixtureRequest):
    path = request.param
    return path, parse(path, source_type=path.parent.name)


def test_document_produces_text(parsed) -> None:
    path, document = parsed

    assert document.text.strip(), f"{path.name} parsed to nothing"
    assert document.segments, f"{path.name} produced no segments"


def test_segments_are_ordered_and_within_bounds(parsed) -> None:
    path, document = parsed

    for segment in document.segments:
        assert 0 <= segment.start < segment.end <= len(document.text)
        assert segment.extraction in constants.EXTRACTION_METHODS

    starts = [segment.start for segment in document.segments]
    assert starts == sorted(starts), f"{path.name} segments out of order"


def test_segment_offsets_match_the_parsed_text(parsed) -> None:
    """A segment's range must actually contain that page's or cell's text."""

    _, document = parsed

    for segment in document.segments:
        assert document.text[segment.start : segment.end].strip()


def test_chunk_offsets_reconstruct_chunks_exactly(parsed) -> None:
    """The invariant every evaluation label depends on, on real documents."""

    path, document = parsed

    chunks = chunk_text(document.text, document.segments)

    assert chunks, f"{path.name} produced no chunks"

    for chunk in chunks:
        assert chunk.text == document.text[chunk.source_start : chunk.source_end], (
            f"{path.name} chunk {chunk.ordinal} offsets do not match its text"
        )


def test_chunks_cover_the_document_without_gaps(parsed) -> None:
    path, document = parsed

    chunks = chunk_text(document.text, document.segments)

    assert chunks[0].source_start == 0
    assert chunks[-1].source_end == len(document.text.rstrip())

    for previous, current in itertools.pairwise(chunks):
        assert current.source_start <= previous.source_end, (
            f"{path.name}: gap between chunks {previous.ordinal} and {current.ordinal}"
        )


def test_every_chunk_carries_provenance(parsed) -> None:
    _, document = parsed

    for chunk in chunk_text(document.text, document.segments):
        assert chunk.extraction in constants.EXTRACTION_METHODS


def test_doc_ids_are_unique_and_slug_safe() -> None:
    ids = [make_doc_id(path) for path in DOCUMENTS]

    assert len(ids) == len(set(ids)), "duplicate doc_id across corpus"

    for doc_id in ids:
        assert doc_id == doc_id.lower()
        assert " " not in doc_id
        assert "/" not in doc_id

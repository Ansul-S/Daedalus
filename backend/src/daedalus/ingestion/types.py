"""
Shared types for the ingestion pipeline.

Every parser produces a ``ParsedDocument``: one flat string of text plus a
record of which region came from where. Chunking then operates on the flat
text and inherits provenance from the segments it overlaps.

Keeping the text flat is what makes character offsets meaningful, and
character offsets are what let evaluation labels survive a change of
chunking strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Chunk", "ParsedDocument", "Segment"]


@dataclass(frozen=True)
class Segment:
    """
    A contiguous region of a document's parsed text.

    One segment per PDF page or per notebook cell, tagged with how its text
    was obtained so that retrieval metrics can be sliced by ingestion path.
    """

    start: int
    end: int
    extraction: str
    page: int | None = None


@dataclass(frozen=True)
class ParsedDocument:
    """A document reduced to text, with provenance."""

    doc_id: str
    text: str
    segments: tuple[Segment, ...]
    source_type: str
    n_pages: int | None = None


@dataclass(frozen=True)
class Chunk:
    """
    A retrievable unit of text.

    Invariant, enforced by tests: ``text == parsed.text[source_start:source_end]``.
    Everything in the evaluation harness depends on it.
    """

    ordinal: int
    text: str
    source_start: int
    source_end: int
    extraction: str
    page: int | None = None

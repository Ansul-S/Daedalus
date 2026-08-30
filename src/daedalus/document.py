"""The canonical representation produced by every parser.

A document is an ordered sequence of segments. Segments carry the heading path
they sit under and a locator identifying where in the original file they came
from, so that anything derived from a segment can cite its source.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class SegmentKind(Enum):
    """The kinds of content a segment can hold.

    These are the categories common to every supported format, rather than
    format-specific cell or block types.
    """

    PROSE = "prose"
    CODE = "code"
    OUTPUT = "output"


@dataclass(frozen=True)
class Segment:
    """One unit of content in its position within a document."""

    ordinal: int
    kind: SegmentKind
    text: str
    heading_path: tuple[str, ...]
    tags: tuple[str, ...]
    locator: str
    parent_ordinal: int | None = None


@dataclass(frozen=True)
class Document:
    """A source file reduced to ordered, locatable segments."""

    doc_id: str
    source_path: Path
    source_format: str
    title: str | None
    segments: tuple[Segment, ...]

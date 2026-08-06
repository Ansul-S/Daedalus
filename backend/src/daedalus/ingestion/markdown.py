"""Markdown parsing — the trivial case, read as-is."""

from __future__ import annotations

from pathlib import Path

from daedalus.config import constants
from daedalus.ingestion.types import ParsedDocument, Segment

__all__ = ["parse_markdown"]


def parse_markdown(path: Path, doc_id: str, source_type: str) -> ParsedDocument:
    """Parse a Markdown file into a single-segment document."""

    text = path.read_text(encoding="utf-8", errors="replace").strip()

    segments = (
        (Segment(start=0, end=len(text), extraction=constants.EXTRACTION_TEXT),) if text else ()
    )

    return ParsedDocument(
        doc_id=doc_id,
        text=text,
        segments=segments,
        source_type=source_type,
    )

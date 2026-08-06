"""
Splitting parsed text into retrievable chunks.

Chunks are produced by walking the text and cutting at the most natural
boundary near each target size — paragraph break first, then line break,
then sentence end, then whitespace. A hard cut is the last resort.

The critical property is that every chunk records exactly where it came
from, such that::

    chunk.text == parsed_text[chunk.source_start:chunk.source_end]

Evaluation labels anchor to character offsets in the parsed text rather
than to chunk IDs, so that re-chunking never invalidates them. That only
works if these offsets are exact.
"""

from __future__ import annotations

from collections.abc import Sequence

from daedalus.config import constants
from daedalus.ingestion.types import Chunk, Segment

__all__ = ["chunk_text"]


# Boundary preference, strongest first. Each entry is a literal to search
# for; the cut is placed immediately after it.
_BOUNDARIES = ("\n\n", "\n", ". ", "? ", "! ", " ")

# A break is only accepted in the last part of the window, so that snapping
# to a boundary cannot produce a chunk far below the target size.
_MIN_FILL = 0.6


def _find_break(text: str, lower: int, upper: int) -> int:
    """
    Return the best cut position in ``[lower, upper)``, or -1 if none.

    The cut lands after the matched boundary, so the delimiter stays with
    the chunk that precedes it.
    """

    for boundary in _BOUNDARIES:
        index = text.rfind(boundary, lower, upper)
        if index != -1:
            return index + len(boundary)

    return -1


def _segment_for(segments: Sequence[Segment], start: int, end: int) -> Segment | None:
    """Return the segment overlapping ``[start, end)`` the most."""

    best: Segment | None = None
    best_overlap = 0

    for segment in segments:
        overlap = min(end, segment.end) - max(start, segment.start)
        if overlap > best_overlap:
            best, best_overlap = segment, overlap

    return best


def chunk_text(
    text: str,
    segments: Sequence[Segment] = (),
    *,
    chunk_size: int = constants.DEFAULT_CHUNK_SIZE,
    overlap: int = constants.DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Split ``text`` into overlapping chunks with exact source offsets.

    ``segments`` supplies provenance; each chunk inherits the extraction
    method and page of whichever segment it overlaps most. An empty
    sequence yields chunks tagged as plain text extraction.
    """

    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")

    length = len(text)
    chunks: list[Chunk] = []
    start = 0
    ordinal = 0

    while start < length:
        # Leading whitespace would be included in the slice and inflate the
        # offsets, so step over it before measuring anything.
        while start < length and text[start].isspace():
            start += 1

        if start >= length:
            break

        limit = min(start + chunk_size, length)

        # Whether this chunk consumes the rest of the text must be decided
        # before trimming: trimming pulls `end` back below `length`, which
        # would otherwise hide the fact that we are done and emit a
        # duplicate final chunk.
        exhausted = limit >= length

        if exhausted:
            end = length
        else:
            window_start = start + int(chunk_size * _MIN_FILL)
            cut = _find_break(text, window_start, limit)
            end = cut if cut != -1 else limit

        # Trailing whitespace is excluded the same way, keeping the stored
        # text identical to the slice its offsets describe.
        while end > start and text[end - 1].isspace():
            end -= 1

        if end <= start:
            start = min(start + chunk_size, length)
            continue

        segment = _segment_for(segments, start, end)

        chunks.append(
            Chunk(
                ordinal=ordinal,
                text=text[start:end],
                source_start=start,
                source_end=end,
                extraction=segment.extraction if segment else constants.EXTRACTION_TEXT,
                page=segment.page if segment else None,
            )
        )
        ordinal += 1

        if exhausted:
            break

        # Step back by the overlap, but never far enough to revisit the
        # previous starting point — that would loop forever.
        start = end - overlap if end - overlap > start else end

    return chunks

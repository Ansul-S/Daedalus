"""
Resolving labelled spans to chunks.

The indirection the whole benchmark rests on. Labels name character ranges
in frozen text; retrieval returns chunk ids. Chunk ids change with every
chunking configuration, and the entire point of the exercise is to compare
chunking configurations — so the two are connected here, at scoring time,
rather than baked into the dataset.

The judgement call is what "this chunk contains the answer" means. A chunk
that clips twenty characters off the end of the relevant sentence has not
retrieved the answer, and counting it would inflate every recall number.
The rule below requires a chunk to hold a real share of the span.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from daedalus.ingestion.types import Chunk

__all__ = ["RelevantSpan", "overlap_fraction", "resolve_spans"]


logger = logging.getLogger(__name__)


# A chunk counts as relevant when it holds at least this share of whichever
# is shorter, the span or the chunk. Comparing against the shorter of the
# two keeps the rule sane at both extremes: a one-sentence span inside a
# large chunk is judged on how much of the sentence survives, and a span
# longer than a chunk is judged on how much of the chunk is inside it.
DEFAULT_MIN_OVERLAP = 0.5


@dataclass(frozen=True)
class RelevantSpan:
    """A labelled character range in a document's frozen text."""

    doc_id: str
    char_start: int
    char_end: int
    grade: int = 2

    def __post_init__(self) -> None:
        if self.char_end <= self.char_start:
            raise ValueError(
                f"span {self.char_start}-{self.char_end} in {self.doc_id!r} is empty or reversed"
            )

    @property
    def length(self) -> int:
        return self.char_end - self.char_start


def overlap_fraction(span: RelevantSpan, chunk: Chunk) -> float:
    """
    How much of the shorter of span and chunk the two share, from 0 to 1.
    """

    overlap = min(span.char_end, chunk.source_end) - max(span.char_start, chunk.source_start)

    if overlap <= 0:
        return 0.0

    shorter = min(span.length, chunk.source_end - chunk.source_start)

    if shorter <= 0:  # pragma: no cover - the schema forbids empty chunks
        return 0.0

    return overlap / shorter


def resolve_spans(
    spans: Iterable[RelevantSpan],
    chunks: Sequence[Chunk],
    chunk_ids: Sequence[int],
    *,
    doc_id: str,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
) -> dict[int, int]:
    """
    Map labelled spans onto the chunk ids that contain them.

    Returns ``{chunk_id: grade}``. A chunk covered by several spans takes
    the highest grade: being partly essential makes it essential, and
    taking the lowest would let a supporting label mask an essential one.

    ``chunks`` and ``chunk_ids`` are parallel — the chunker produces the
    former, the database assigns the latter.
    """

    if len(chunks) != len(chunk_ids):
        raise ValueError(f"{len(chunks)} chunks but {len(chunk_ids)} ids")

    grades: dict[int, int] = {}

    for span in spans:
        if span.doc_id != doc_id:
            continue

        matched = False

        for chunk, chunk_id in zip(chunks, chunk_ids, strict=True):
            if overlap_fraction(span, chunk) >= min_overlap:
                grades[chunk_id] = max(grades.get(chunk_id, 0), span.grade)
                matched = True

        if not matched:
            # Worth surfacing: a span no chunk covers is unreachable, so
            # every query using it is unanswerable no matter how good
            # retrieval is, and its recall is capped below 1 for reasons
            # that have nothing to do with the retriever.
            logger.warning(
                "Span %s:%d-%d resolves to no chunk at overlap >= %.2f",
                span.doc_id,
                span.char_start,
                span.char_end,
                min_overlap,
            )

    return grades

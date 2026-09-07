"""Building the bounded context a question is generated from.

The algorithm is fixed by `docs/PHASE-6-PROTOCOL.md` section 3 and committed
before any question was generated:

    1. Seed = the longest prose chunk, ties broken by lowest ordinal.
    2. All prose chunks, in ordinal order. Prose is never truncated or dropped.
    3. Code and output in ordinal order under a 2,000-character budget. The
       chunk that crosses the budget is truncated and marked; everything after
       it is omitted and counted.
    4. Every chunk is labelled with its (doc_id, ordinal); the seed is marked.

Prose is exempt from the budget because it can afford to be: the largest prose
in any eligible section of this corpus measures 2,686 characters, so including
all of it bounds the context at 4,686 characters without a rule.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import cast

import psycopg

Connection = psycopg.Connection[tuple[object, ...]]

#: Characters of code and output admitted per section. Protocol section 3.
NONPROSE_BUDGET_CHARS = 2000

#: Appended to the one chunk that crosses the budget.
TRUNCATION_MARKER = "[truncated]"

#: Chunk kinds that the budget applies to, in the order they are admitted.
BUDGETED_KINDS = ("code", "output")

_LOAD_SECTION_CHUNKS = """
SELECT ordinal, kind, text
FROM chunks
WHERE doc_id = %s AND heading_path = %s
ORDER BY ordinal
"""


@dataclass(frozen=True)
class SourceChunk:
    """A chunk of a section as stored, before any budget is applied."""

    ordinal: int
    kind: str
    text: str


@dataclass(frozen=True)
class ContextChunk:
    """A chunk as it will be shown to the generator."""

    ordinal: int
    kind: str
    text: str
    is_seed: bool
    truncated: bool


@dataclass(frozen=True)
class SectionContext:
    """Everything the generator is shown for one section, and what was left out."""

    doc_id: str
    heading_path: tuple[str, ...]
    seed_ordinal: int
    chunks: tuple[ContextChunk, ...]
    omitted_chunks: int
    omitted_chars: int

    @property
    def prose_chars(self) -> int:
        """Characters of prose included. Never reduced by the budget."""
        return sum(len(c.text) for c in self.chunks if c.kind == "prose")

    @property
    def nonprose_chars(self) -> int:
        """Characters of code and output included, after any truncation."""
        return sum(len(c.text) for c in self.chunks if c.kind != "prose")

    @property
    def total_chars(self) -> int:
        """Characters of chunk text in the context, markers excluded."""
        return sum(len(c.text) for c in self.chunks)


class NoProseError(ValueError):
    """Raised when a section has no prose and so has no seed."""


def load_section_chunks(
    connection: Connection, doc_id: str, heading_path: Sequence[str]
) -> list[SourceChunk]:
    """Read one section's chunks in ordinal order."""
    with connection.cursor() as cursor:
        cursor.execute(_LOAD_SECTION_CHUNKS, (doc_id, list(heading_path)))
        return [
            SourceChunk(
                ordinal=cast("int", row[0]),
                kind=cast("str", row[1]),
                text=cast("str", row[2]),
            )
            for row in cursor.fetchall()
        ]


def choose_seed(chunks: Iterable[SourceChunk]) -> int:
    """Return the ordinal of the section's seed chunk.

    The longest prose chunk, ties broken by lowest ordinal. Length is the
    proxy for substance; the tie-break exists so the choice is reproducible
    rather than dependent on row order.
    """
    prose = [chunk for chunk in chunks if chunk.kind == "prose"]
    if not prose:
        raise NoProseError("section has no prose chunk and therefore no seed")
    return min(prose, key=lambda chunk: (-len(chunk.text), chunk.ordinal)).ordinal


def build_context(
    doc_id: str,
    heading_path: Sequence[str],
    chunks: Sequence[SourceChunk],
    budget: int = NONPROSE_BUDGET_CHARS,
) -> SectionContext:
    """Assemble the bounded context for one section.

    All prose is included whatever its length. Code and output are admitted in
    ordinal order until the budget runs out; the chunk that crosses it is
    truncated, and every chunk after it is omitted and counted so that what was
    dropped is visible rather than silent.
    """
    if budget < 0:
        raise ValueError(f"budget cannot be negative, got {budget}")

    seed_ordinal = choose_seed(chunks)

    included: list[ContextChunk] = [
        ContextChunk(
            ordinal=chunk.ordinal,
            kind=chunk.kind,
            text=chunk.text,
            is_seed=chunk.ordinal == seed_ordinal,
            truncated=False,
        )
        for chunk in sorted(chunks, key=lambda c: c.ordinal)
        if chunk.kind == "prose"
    ]

    remaining = budget
    omitted_chunks = 0
    omitted_chars = 0
    exhausted = False

    for chunk in sorted(chunks, key=lambda c: c.ordinal):
        if chunk.kind not in BUDGETED_KINDS:
            continue
        if exhausted or remaining <= 0:
            omitted_chunks += 1
            omitted_chars += len(chunk.text)
            continue
        if len(chunk.text) <= remaining:
            included.append(
                ContextChunk(chunk.ordinal, chunk.kind, chunk.text, False, False)
            )
            remaining -= len(chunk.text)
            continue

        included.append(
            ContextChunk(chunk.ordinal, chunk.kind, chunk.text[:remaining], False, True)
        )
        omitted_chars += len(chunk.text) - remaining
        remaining = 0
        exhausted = True

    included.sort(key=lambda c: c.ordinal)
    return SectionContext(
        doc_id=doc_id,
        heading_path=tuple(heading_path),
        seed_ordinal=seed_ordinal,
        chunks=tuple(included),
        omitted_chunks=omitted_chunks,
        omitted_chars=omitted_chars,
    )


def section_context(
    connection: Connection,
    doc_id: str,
    heading_path: Sequence[str],
    budget: int = NONPROSE_BUDGET_CHARS,
) -> SectionContext:
    """Load a section and build its bounded context."""
    return build_context(
        doc_id,
        heading_path,
        load_section_chunks(connection, doc_id, heading_path),
        budget,
    )


def render_context(context: SectionContext) -> str:
    """Render the context block shown to the generator.

    Each chunk carries its (doc_id, ordinal) so a citation can be checked
    against what was actually supplied, and the seed is marked so the model is
    told which chunk the question must be about. This renders the context block
    only; the instructions around it belong to the prompt.
    """
    lines = [f"SECTION: {' > '.join(context.heading_path)}", ""]
    for chunk in context.chunks:
        marker = " (SEED)" if chunk.is_seed else ""
        label = f"[{context.doc_id}:{chunk.ordinal}] {chunk.kind}{marker}"
        body = chunk.text + (f" {TRUNCATION_MARKER}" if chunk.truncated else "")
        lines.extend((label, body, ""))
    return "\n".join(lines).rstrip() + "\n"


def context_ordinals(context: SectionContext) -> tuple[int, ...]:
    """Return every ordinal supplied, for checking citations against."""
    return tuple(chunk.ordinal for chunk in context.chunks)

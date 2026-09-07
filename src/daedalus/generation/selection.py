"""Section eligibility, the frozen 300-section draw, and its type mapping.

Every constant here is fixed by `docs/PHASE-6-PROTOCOL.md`, committed before any
question was generated. Changing one changes which sections are generated from
and invalidates the pre-registration, so they are module constants with no
command-line override, guarded by tests.

The functions that decide anything take plain `Section` values rather than a
connection, so the draw can be reproduced and tested without a database.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import cast

import psycopg

Connection = psycopg.Connection[tuple[object, ...]]

#: Minimum prose characters for a section to be eligible. Protocol section 2.
MIN_PROSE_CHARS = 300

#: Prose at or above which a section is asked for a comparison. Section 5.
COMPARISON_PROSE_CHARS = 1000

#: Prose at or above which a section is asked for an explanation. Section 5.
EXPLANATION_PROSE_CHARS = 600

#: Seed string for the selection hash. Section 4. Part of the pre-registration.
SELECTION_SEED = "phase6-selection-20260907"

#: How many eligible sections are generated from. Section 4.
SELECTION_SIZE = 300

#: Unit separator joining the parts of a section key. Chosen because it cannot
#: occur in a heading, so no two distinct sections can collide on a key.
KEY_SEPARATOR = "\x1f"

#: The three eligibility classes. They partition every section exactly.
ELIGIBLE = "eligible"
THIN = "thin"
PROSE_FREE = "prose_free"

#: Difficulty levels in the order the rotation assigns them. Section 6.
DIFFICULTY_CYCLE = ("easy", "medium", "hard")

_LOAD_SECTIONS = """
SELECT doc_id,
       heading_path,
       coalesce(sum(length(text)) FILTER (WHERE kind = 'prose'), 0),
       count(*) FILTER (WHERE kind = 'code'),
       count(*)
FROM chunks
GROUP BY doc_id, heading_path
"""


@dataclass(frozen=True)
class Section:
    """One heading section, reduced to the properties the protocol reads."""

    doc_id: str
    heading_path: tuple[str, ...]
    prose_chars: int
    code_chunks: int
    total_chunks: int


@dataclass(frozen=True)
class SelectedSection:
    """An eligible section drawn into the benchmark, with what it will be asked."""

    section: Section
    rank: int
    question_type: str
    difficulty: str


@dataclass(frozen=True)
class Partition:
    """Every section sorted into exactly one eligibility class."""

    eligible: tuple[Section, ...]
    thin: tuple[Section, ...]
    prose_free: tuple[Section, ...]

    @property
    def total(self) -> int:
        """The number of sections partitioned."""
        return len(self.eligible) + len(self.thin) + len(self.prose_free)


def load_sections(connection: Connection) -> list[Section]:
    """Read every heading section in the store, in no particular order.

    Ordering is deliberately not imposed here. The draw derives its own total
    order from the section key, so a caller cannot influence the sample by
    handing sections over in a different sequence.
    """
    with connection.cursor() as cursor:
        cursor.execute(_LOAD_SECTIONS)
        return [
            Section(
                doc_id=cast("str", row[0]),
                heading_path=tuple(cast("list[str]", row[1])),
                prose_chars=cast("int", row[2]),
                code_chunks=cast("int", row[3]),
                total_chunks=cast("int", row[4]),
            )
            for row in cursor.fetchall()
        ]


def eligibility(section: Section) -> str:
    """Return which of the three eligibility classes a section falls in."""
    if section.prose_chars == 0:
        return PROSE_FREE
    if section.prose_chars < MIN_PROSE_CHARS:
        return THIN
    return ELIGIBLE


def partition(sections: Iterable[Section]) -> Partition:
    """Sort sections into the three classes, preserving nothing but membership.

    Prose-free and thin sections are kept rather than discarded: both are
    reported alongside the coverage figure, and a denominator no one can see the
    exclusions from is not auditable.
    """
    classes: dict[str, list[Section]] = {ELIGIBLE: [], THIN: [], PROSE_FREE: []}
    for section in sections:
        classes[eligibility(section)].append(section)
    return Partition(
        eligible=tuple(classes[ELIGIBLE]),
        thin=tuple(classes[THIN]),
        prose_free=tuple(classes[PROSE_FREE]),
    )


def section_key(section: Section) -> str:
    """Return the string identifying a section for hashing.

    Matches `doc_id || CHR(31) || array_to_string(heading_path, CHR(31))` so the
    Python draw and the SQL the protocol was written from agree exactly.
    """
    return KEY_SEPARATOR.join((section.doc_id, *section.heading_path))


def selection_hash(section: Section, seed: str = SELECTION_SEED) -> str:
    """Return the hash a section is ordered by in the draw."""
    return hashlib.md5(f"{section_key(section)}:{seed}".encode()).hexdigest()


def question_type_for(section: Section) -> str:
    """Return the question type a section is asked for.

    First match wins, so the four types are disjoint. Code presence is checked
    before prose length because a section holding code can be asked to reason
    about it whatever its prose length, whereas the prose thresholds only ever
    stand in for how much there is to say.
    """
    if section.code_chunks > 0:
        return "code_reasoning"
    if section.prose_chars >= COMPARISON_PROSE_CHARS:
        return "comparison"
    if section.prose_chars >= EXPLANATION_PROSE_CHARS:
        return "explanation"
    return "conceptual"


def difficulty_for(rank: int) -> str:
    """Return the difficulty requested at a given position in the draw.

    Rotating on rank makes difficulty independent of every section property.
    Matching difficulty to how rich a section is would confound the instruction
    with the material, leaving no way to tell whether a question was judged hard
    because it was generated hard or because its source was dense.
    """
    if rank < 1:
        raise ValueError(f"rank is 1-based, got {rank}")
    return DIFFICULTY_CYCLE[(rank - 1) % len(DIFFICULTY_CYCLE)]


def draw_order(
    sections: Iterable[Section], seed: str = SELECTION_SEED
) -> list[Section]:
    """Return every given section in the frozen draw order.

    Ordered by hash, then by doc_id and heading path. The tie-break makes the
    order total: without it two sections colliding on a hash would be left in
    whatever order they arrived in, and the draw would stop being reproducible.
    """
    return sorted(
        sections,
        key=lambda section: (
            selection_hash(section, seed),
            section.doc_id,
            section.heading_path,
        ),
    )


def select(
    sections: Iterable[Section],
    size: int = SELECTION_SIZE,
    seed: str = SELECTION_SEED,
) -> list[SelectedSection]:
    """Draw the benchmark sample from sections already known to be eligible.

    Eligibility is not applied here. Passing sections that were never filtered
    would silently draw thin and prose-free material into the benchmark, so the
    caller states which sections are eligible and this function draws from them.
    """
    if size < 0:
        raise ValueError(f"size cannot be negative, got {size}")

    ordered = draw_order(sections, seed)
    if size > len(ordered):
        raise ValueError(f"asked for {size} sections but only {len(ordered)} eligible")

    return [
        SelectedSection(
            section=section,
            rank=rank,
            question_type=question_type_for(section),
            difficulty=difficulty_for(rank),
        )
        for rank, section in enumerate(ordered[:size], start=1)
    ]


def type_counts(selected: Sequence[SelectedSection]) -> dict[str, int]:
    """Return how many of a selection carry each question type."""
    counts: dict[str, int] = {}
    for item in selected:
        counts[item.question_type] = counts.get(item.question_type, 0) + 1
    return counts


def difficulty_counts(selected: Sequence[SelectedSection]) -> dict[str, int]:
    """Return how many of a selection carry each requested difficulty."""
    counts: dict[str, int] = {}
    for item in selected:
        counts[item.difficulty] = counts.get(item.difficulty, 0) + 1
    return counts

"""Tests for the frozen section draw, its type mapping and its difficulty rota.

The counts asserted here are computed from synthetic sections, not from the
corpus. A test that depended on the ingested notebooks would fail on a checkout
without them, and `corpus/` is untracked. What the corpus actually produces is
verified separately against the numbers recorded in docs/PHASE-6-PROTOCOL.md.
"""

from __future__ import annotations

from pathlib import Path

import psycopg
import pytest

from daedalus.document import Document, Segment, SegmentKind
from daedalus.generation.selection import (
    COMPARISON_PROSE_CHARS,
    DIFFICULTY_CYCLE,
    ELIGIBLE,
    EXPLANATION_PROSE_CHARS,
    KEY_SEPARATOR,
    MIN_PROSE_CHARS,
    PROSE_FREE,
    SELECTION_SEED,
    SELECTION_SIZE,
    THIN,
    Section,
    difficulty_counts,
    difficulty_for,
    draw_order,
    eligibility,
    load_sections,
    partition,
    question_type_for,
    section_key,
    select,
    selection_hash,
    type_counts,
)
from daedalus.storage.documents import store_document

Connection = psycopg.Connection[tuple[object, ...]]


def build_section(
    name: str = "s",
    prose_chars: int = 500,
    code_chunks: int = 0,
    doc_id: str = "doc",
) -> Section:
    """A section with only the properties the protocol reads."""
    return Section(
        doc_id=doc_id,
        heading_path=(name,),
        prose_chars=prose_chars,
        code_chunks=code_chunks,
        total_chunks=1 + code_chunks,
    )


def corpus_shaped_sections() -> list[Section]:
    """408 synthetic sections partitioning 337 / 56 / 15, as the corpus does.

    The shape mirrors the measured corpus so the partition and draw are
    exercised at the size they will actually run at. The text is synthetic; only
    the counts are borrowed.
    """
    sections = []
    for i in range(15):
        sections.append(build_section(f"free-{i}", prose_chars=0))
    for i in range(56):
        sections.append(build_section(f"thin-{i}", prose_chars=100 + i))
    for i in range(337):
        sections.append(build_section(f"ok-{i}", prose_chars=MIN_PROSE_CHARS + i))
    return sections


def test_the_frozen_constants_are_what_the_protocol_declares() -> None:
    """A guard on the pre-registration, not a test of behaviour.

    These values are quoted in docs/PHASE-6-PROTOCOL.md and fix which sections
    are generated from. If one changes, the draw changes and the committed
    protocol no longer describes the experiment.
    """
    assert MIN_PROSE_CHARS == 300
    assert EXPLANATION_PROSE_CHARS == 600
    assert COMPARISON_PROSE_CHARS == 1000
    assert SELECTION_SEED == "phase6-selection-20260907"
    assert SELECTION_SIZE == 300
    assert DIFFICULTY_CYCLE == ("easy", "medium", "hard")


@pytest.mark.parametrize(
    ("prose", "expected"),
    [
        (0, PROSE_FREE),
        (1, THIN),
        (299, THIN),
        (300, ELIGIBLE),
        (5000, ELIGIBLE),
    ],
)
def test_eligibility_boundaries(prose: int, expected: str) -> None:
    assert eligibility(build_section(prose_chars=prose)) == expected


def test_the_three_classes_partition_every_section() -> None:
    result = partition(corpus_shaped_sections())

    assert (len(result.eligible), len(result.thin), len(result.prose_free)) == (
        337,
        56,
        15,
    )
    assert result.total == 408


def test_the_draw_takes_300_of_the_337_eligible() -> None:
    eligible = partition(corpus_shaped_sections()).eligible

    selected = select(eligible)

    assert len(selected) == SELECTION_SIZE
    assert len({s.section.heading_path for s in selected}) == SELECTION_SIZE
    assert [s.rank for s in selected] == list(range(1, 301))


def test_the_draw_assigns_difficulty_in_equal_thirds() -> None:
    eligible = partition(corpus_shaped_sections()).eligible

    assert difficulty_counts(select(eligible)) == {
        "easy": 100,
        "medium": 100,
        "hard": 100,
    }


def test_the_draw_is_independent_of_the_order_sections_arrive_in() -> None:
    eligible = list(partition(corpus_shaped_sections()).eligible)

    forwards = select(eligible)
    backwards = select(list(reversed(eligible)))

    assert [s.section for s in forwards] == [s.section for s in backwards]


def test_the_draw_is_stable_across_calls() -> None:
    eligible = partition(corpus_shaped_sections()).eligible

    assert [s.section for s in select(eligible)] == [
        s.section for s in select(eligible)
    ]


def test_a_different_seed_draws_a_different_sample() -> None:
    eligible = partition(corpus_shaped_sections()).eligible

    default = [s.section for s in select(eligible)]
    other = [s.section for s in select(eligible, seed="something-else")]

    assert default != other


def test_asking_for_more_than_is_eligible_raises() -> None:
    with pytest.raises(ValueError, match="only 3 eligible"):
        select([build_section("a"), build_section("b"), build_section("c")], size=4)


@pytest.mark.parametrize(
    ("prose", "code", "expected"),
    [
        (300, 1, "code_reasoning"),
        (5000, 2, "code_reasoning"),
        (1000, 0, "comparison"),
        (999, 0, "explanation"),
        (600, 0, "explanation"),
        (599, 0, "conceptual"),
        (300, 0, "conceptual"),
    ],
)
def test_question_type_mapping(prose: int, code: int, expected: str) -> None:
    assert question_type_for(build_section(prose_chars=prose, code_chunks=code)) == (
        expected
    )


def test_code_presence_outranks_prose_length() -> None:
    """A long section holding code is asked to reason about the code."""
    assert question_type_for(build_section(prose_chars=9000, code_chunks=1)) == (
        "code_reasoning"
    )


def test_every_type_and_difficulty_is_reachable_in_one_draw() -> None:
    sections = [build_section(f"concept-{i}", prose_chars=400) for i in range(150)]
    sections += [build_section(f"explain-{i}", prose_chars=700) for i in range(90)]
    sections += [build_section(f"compare-{i}", prose_chars=1200) for i in range(40)]
    sections += [
        build_section(f"code-{i}", prose_chars=400, code_chunks=1) for i in range(60)
    ]

    counts = type_counts(select(sections))

    assert set(counts) == {
        "conceptual",
        "explanation",
        "comparison",
        "code_reasoning",
    }
    assert sum(counts.values()) == SELECTION_SIZE


@pytest.mark.parametrize(
    ("rank", "expected"),
    [(1, "easy"), (2, "medium"), (3, "hard"), (4, "easy"), (300, "hard")],
)
def test_difficulty_rotates_on_rank(rank: int, expected: str) -> None:
    assert difficulty_for(rank) == expected


def test_difficulty_rejects_a_rank_below_one() -> None:
    with pytest.raises(ValueError, match="1-based"):
        difficulty_for(0)


def test_the_section_key_separates_parts_unambiguously() -> None:
    section = Section("doc", ("A", "B"), 400, 0, 1)

    assert section_key(section) == f"doc{KEY_SEPARATOR}A{KEY_SEPARATOR}B"


def test_sections_differing_only_in_heading_split_hash_differently() -> None:
    """A separator that could appear in a heading would collide these two."""
    one = Section("doc", ("A", "B"), 400, 0, 1)
    two = Section("doc", ("A B",), 400, 0, 1)

    assert selection_hash(one) != selection_hash(two)


def test_draw_order_is_by_hash_not_by_name() -> None:
    sections = [build_section(f"s-{i}") for i in range(20)]

    ordered = draw_order(sections)

    assert ordered != sections
    assert sorted(ordered, key=section_key) == sorted(sections, key=section_key)


def test_load_sections_reads_prose_and_code_per_section(
    connection: Connection,
) -> None:
    store_document(
        connection,
        Document(
            doc_id="d1",
            source_path=Path("/corpus/n.ipynb"),
            source_format="notebook",
            title="n",
            segments=(
                Segment(0, SegmentKind.PROSE, "a" * 400, ("S1",), (), "cell:0"),
                Segment(1, SegmentKind.CODE, "x = 1", ("S1",), (), "cell:1"),
                Segment(2, SegmentKind.PROSE, "b" * 50, ("S2",), (), "cell:2"),
            ),
        ),
    )

    by_path = {s.heading_path: s for s in load_sections(connection)}

    assert by_path[("S1",)].prose_chars == 400
    assert by_path[("S1",)].code_chunks == 1
    assert by_path[("S1",)].total_chunks == 2
    assert by_path[("S2",)].prose_chars == 50
    assert eligibility(by_path[("S2",)]) == THIN

"""Tests for conversion into the canonical document representation.

These specify the behaviour the functions in `canonical.py` must produce.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest

from daedalus.document import SegmentKind
from daedalus.ingestion.canonical import (
    SOURCE_FORMAT,
    TRUNCATION_MARKER,
    extract_title,
    notebook_to_document,
    stable_document_id,
)
from daedalus.ingestion.notebook import MAX_OUTPUT_CHARS, parse_notebook
from tests.test_notebook import code, md, result, stream, write_notebook


def convert(tmp_path: Path, cells: list[nbformat.NotebookNode], name: str = "n.ipynb"):
    return notebook_to_document(parse_notebook(write_notebook(tmp_path / name, cells)))


def raw(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_raw_cell(source)


def test_document_id_depends_only_on_content() -> None:
    assert stable_document_id("same text") == stable_document_id("same text")
    assert stable_document_id("a") != stable_document_id("b")


def test_document_id_is_sixteen_hex_characters() -> None:
    value = stable_document_id("anything")
    assert len(value) == 16
    assert all(c in "0123456789abcdef" for c in value)


def test_extract_title_returns_outermost_first_heading(tmp_path: Path) -> None:
    parsed = parse_notebook(
        write_notebook(tmp_path / "n.ipynb", [md("# Section 1"), md("## 1.1 Detail")])
    )
    assert extract_title(parsed) == "Section 1"


def test_extract_title_returns_none_without_headings(tmp_path: Path) -> None:
    parsed = parse_notebook(
        write_notebook(tmp_path / "n.ipynb", [md("just prose"), code("x = 1")])
    )
    assert extract_title(parsed) is None


def test_document_carries_the_title(tmp_path: Path) -> None:
    doc = convert(tmp_path, [md("# Section 1"), md("## 1.1 Detail")])
    assert doc.title == "Section 1"


def test_markdown_becomes_a_prose_segment(tmp_path: Path) -> None:
    doc = convert(tmp_path, [md("# A\n\nbody text")])
    assert len(doc.segments) == 1
    segment = doc.segments[0]
    assert segment.kind is SegmentKind.PROSE
    assert segment.text == "# A\n\nbody text"
    assert segment.heading_path == ("A",)
    assert segment.parent_ordinal is None


def test_code_and_outputs_become_linked_segments(tmp_path: Path) -> None:
    doc = convert(tmp_path, [md("# A"), code("print(1)", [stream("1\n")])])

    prose, source, output = doc.segments
    assert (prose.kind, source.kind, output.kind) == (
        SegmentKind.PROSE,
        SegmentKind.CODE,
        SegmentKind.OUTPUT,
    )
    assert source.text == "print(1)"
    assert output.text == "1\n"
    assert output.parent_ordinal == source.ordinal


def test_multiple_outputs_share_one_parent(tmp_path: Path) -> None:
    doc = convert(
        tmp_path,
        [code("f()", [stream("first\n"), result({"text/plain": "second"})])],
    )

    source, first, second = doc.segments
    assert first.parent_ordinal == source.ordinal
    assert second.parent_ordinal == source.ordinal
    assert [first.text, second.text] == ["first\n", "second"]


def test_ordinals_are_contiguous_across_segments(tmp_path: Path) -> None:
    doc = convert(
        tmp_path,
        [md("# A"), code("a()", [stream("x")]), md("more"), code("b()")],
    )
    assert [s.ordinal for s in doc.segments] == list(range(len(doc.segments)))


def test_blank_cells_are_skipped(tmp_path: Path) -> None:
    doc = convert(tmp_path, [md("# A"), md("   \n"), code(""), code("x = 1")])
    assert [s.text for s in doc.segments] == ["# A", "x = 1"]


def test_locator_identifies_the_source_cell(tmp_path: Path) -> None:
    doc = convert(tmp_path, [md("# A"), md("   "), code("x = 1", [stream("out")])])

    # the blank cell is skipped, but locators still reference notebook indices
    assert [s.locator for s in doc.segments] == ["cell:0", "cell:2", "cell:2"]


def test_tags_and_headings_carry_onto_outputs(tmp_path: Path) -> None:
    doc = convert(
        tmp_path,
        [md("# A"), md("## Concept Check"), code("q()", [stream("answer")])],
    )
    output = doc.segments[-1]
    assert output.heading_path == ("A", "Concept Check")
    assert output.tags == ()


def test_tagged_markdown_keeps_its_tag(tmp_path: Path) -> None:
    doc = convert(tmp_path, [md("## Instructor-Only Answers\n\nbody")])
    assert doc.segments[0].tags == ("instructor-answers",)


def test_truncated_output_is_marked(tmp_path: Path) -> None:
    doc = convert(tmp_path, [code("big()", [stream("x" * (MAX_OUTPUT_CHARS + 10))])])
    output = doc.segments[-1]
    assert output.text.endswith(TRUNCATION_MARKER)


def test_document_metadata_is_populated(tmp_path: Path) -> None:
    doc = convert(tmp_path, [md("# A"), code("x = 1")], name="notes.ipynb")

    assert doc.source_format == SOURCE_FORMAT
    assert doc.source_path.name == "notes.ipynb"
    assert doc.doc_id == stable_document_id("# A\nx = 1")


def test_same_content_different_path_gives_same_id(tmp_path: Path) -> None:
    cells = [md("# A"), code("x = 1")]
    first = convert(tmp_path, cells, name="one.ipynb")
    second = convert(tmp_path, cells, name="two.ipynb")
    assert first.doc_id == second.doc_id


@pytest.mark.parametrize("kind", [SegmentKind.PROSE, SegmentKind.CODE])
def test_non_output_segments_have_no_parent(tmp_path: Path, kind: SegmentKind) -> None:
    doc = convert(tmp_path, [md("# A"), code("x = 1", [stream("out")])])
    assert all(s.parent_ordinal is None for s in doc.segments if s.kind is kind)


def test_blank_cells_do_not_affect_document_id(tmp_path: Path) -> None:
    without = convert(tmp_path, [md("# A"), code("x = 1")], name="a.ipynb")
    with_blank = convert(
        tmp_path, [md("# A"), md("   \n"), code("x = 1")], name="b.ipynb"
    )
    assert without.doc_id == with_blank.doc_id


def test_raw_cells_produce_no_segments(tmp_path: Path) -> None:
    doc = convert(tmp_path, [md("# A"), raw("\\usepackage{amsmath}"), code("x = 1")])
    assert [s.kind for s in doc.segments] == [SegmentKind.PROSE, SegmentKind.CODE]


def test_raw_cells_do_not_affect_document_id(tmp_path: Path) -> None:
    without = convert(tmp_path, [md("# A"), code("x = 1")], name="a.ipynb")
    with_raw = convert(
        tmp_path, [md("# A"), raw("preamble"), code("x = 1")], name="b.ipynb"
    )
    assert without.doc_id == with_raw.doc_id

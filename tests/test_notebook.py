"""Tests for notebook parsing.

Notebooks are built in-test rather than read from the corpus, which is not
tracked in version control.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest

from daedalus.ingestion.notebook import (
    MAX_OUTPUT_CHARS,
    parse_notebook,
)


def write_notebook(path: Path, cells: list[nbformat.NotebookNode]) -> Path:
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"name": "python3"}
    notebook.metadata["language_info"] = {"name": "python"}
    nbformat.write(notebook, str(path))
    return path


def md(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source)


def code(
    source: str, outputs: list[nbformat.NotebookNode] | None = None
) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source, outputs=outputs or [])


def stream(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_output("stream", name="stdout", text=text)


def result(data: dict[str, object]) -> nbformat.NotebookNode:
    return nbformat.v4.new_output("execute_result", data=data)


def test_reads_cells_and_kernel_metadata(tmp_path: Path) -> None:
    path = write_notebook(tmp_path / "n.ipynb", [md("# Title"), code("x = 1")])

    parsed = parse_notebook(path)

    assert len(parsed.cells) == 2
    assert parsed.cells[0].cell_type == "markdown"
    assert parsed.cells[1].source == "x = 1"
    assert parsed.kernel_name == "python3"
    assert parsed.language == "python"


def test_cell_is_filed_under_its_own_heading(tmp_path: Path) -> None:
    path = write_notebook(tmp_path / "n.ipynb", [md("# Section 2\n\nIntro text.")])

    parsed = parse_notebook(path)

    assert parsed.cells[0].heading_path == ("Section 2",)


def test_nested_headings_build_a_path(tmp_path: Path) -> None:
    path = write_notebook(
        tmp_path / "n.ipynb",
        [md("# Section 2"), md("## 2.1 The Core Question"), md("### Detail")],
    )

    parsed = parse_notebook(path)

    assert parsed.cells[0].heading_path == ("Section 2",)
    assert parsed.cells[1].heading_path == ("Section 2", "2.1 The Core Question")
    assert parsed.cells[2].heading_path == (
        "Section 2",
        "2.1 The Core Question",
        "Detail",
    )


def test_sibling_heading_replaces_previous(tmp_path: Path) -> None:
    path = write_notebook(
        tmp_path / "n.ipynb", [md("# A"), md("## A.1"), md("## A.2"), md("# B")]
    )

    parsed = parse_notebook(path)

    assert parsed.cells[2].heading_path == ("A", "A.2")
    assert parsed.cells[3].heading_path == ("B",)


def test_code_cell_inherits_current_heading_path(tmp_path: Path) -> None:
    path = write_notebook(
        tmp_path / "n.ipynb", [md("# A"), md("## A.1"), code("x = 1")]
    )

    parsed = parse_notebook(path)

    assert parsed.cells[2].heading_path == ("A", "A.1")


def test_multiple_headings_in_one_cell(tmp_path: Path) -> None:
    path = write_notebook(
        tmp_path / "n.ipynb", [md("# A\n\ntext\n\n## A.1\n\nmore"), code("x = 1")]
    )

    parsed = parse_notebook(path)

    assert parsed.cells[0].heading_path == ("A",)
    assert parsed.cells[1].heading_path == ("A", "A.1")


def test_markdown_without_heading_keeps_current_path(tmp_path: Path) -> None:
    path = write_notebook(tmp_path / "n.ipynb", [md("# A"), md("Just prose.")])

    parsed = parse_notebook(path)

    assert parsed.cells[1].heading_path == ("A",)


def test_hash_without_space_is_not_a_heading(tmp_path: Path) -> None:
    path = write_notebook(tmp_path / "n.ipynb", [md("# A"), md("#hashtag not a head")])

    parsed = parse_notebook(path)

    assert parsed.cells[1].heading_path == ("A",)


def test_stream_output_is_kept(tmp_path: Path) -> None:
    path = write_notebook(tmp_path / "n.ipynb", [code("print(1)", [stream("1\n")])])

    parsed = parse_notebook(path)

    assert [o.text for o in parsed.cells[0].outputs] == ["1\n"]
    assert parsed.cells[0].dropped_outputs == ()


def test_plain_text_kept_and_other_mimetypes_recorded_as_dropped(
    tmp_path: Path,
) -> None:
    path = write_notebook(
        tmp_path / "n.ipynb",
        [
            code(
                "df",
                [
                    result(
                        {
                            "text/plain": "   a  b",
                            "text/html": "<table></table>",
                            "application/vnd.jupyter.widget-view+json": {"v": 1},
                        }
                    )
                ],
            )
        ],
    )

    parsed = parse_notebook(path)
    cell = parsed.cells[0]

    assert [o.text for o in cell.outputs] == ["   a  b"]
    assert set(cell.dropped_outputs) == {
        "execute_result:text/html",
        "execute_result:application/vnd.jupyter.widget-view+json",
    }


def test_output_without_kept_mimetype_yields_no_output(tmp_path: Path) -> None:
    path = write_notebook(
        tmp_path / "n.ipynb",
        [code("bar", [result({"application/vnd.jupyter.widget-view+json": {}})])],
    )

    parsed = parse_notebook(path)

    assert parsed.cells[0].outputs == ()
    assert parsed.cells[0].dropped_outputs == (
        "execute_result:application/vnd.jupyter.widget-view+json",
    )


def test_long_output_is_truncated(tmp_path: Path) -> None:
    path = write_notebook(
        tmp_path / "n.ipynb", [code("big", [stream("x" * (MAX_OUTPUT_CHARS + 50))])]
    )

    parsed = parse_notebook(path)
    output = parsed.cells[0].outputs[0]

    assert output.truncated is True
    assert len(output.text) == MAX_OUTPUT_CHARS


def test_error_output_is_kept_as_text(tmp_path: Path) -> None:
    error = nbformat.v4.new_output(
        "error", ename="ValueError", evalue="bad input", traceback=["..."]
    )
    path = write_notebook(tmp_path / "n.ipynb", [code("boom()", [error])])

    parsed = parse_notebook(path)

    assert parsed.cells[0].outputs[0].text == "ValueError: bad input"


def test_unexecuted_code_cell_has_no_outputs(tmp_path: Path) -> None:
    path = write_notebook(tmp_path / "n.ipynb", [code("x = 1")])

    parsed = parse_notebook(path)

    assert parsed.cells[0].outputs == ()
    assert parsed.cells[0].execution_count is None


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("## Instructor-Only Answers", ("instructor-answers",)),
        ("## Concept Check", ("concept-check",)),
        ("## 1.8 Thought Experiment", ("thought-experiment",)),
        ("## 1.6 The Main Limitation", ()),
    ],
)
def test_pedagogical_headings_are_tagged(
    tmp_path: Path, heading: str, expected: tuple[str, ...]
) -> None:
    path = write_notebook(tmp_path / "n.ipynb", [md(f"{heading}\n\nbody")])

    parsed = parse_notebook(path)

    assert parsed.cells[0].tags == expected


def test_tags_do_not_leak_to_following_cells(tmp_path: Path) -> None:
    path = write_notebook(
        tmp_path / "n.ipynb", [md("## Concept Check"), md("plain prose"), code("x = 1")]
    )

    parsed = parse_notebook(path)

    assert parsed.cells[0].tags == ("concept-check",)
    assert parsed.cells[1].tags == ()
    assert parsed.cells[2].tags == ()


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_notebook(tmp_path / "absent.ipynb")

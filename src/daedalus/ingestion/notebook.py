"""Parsing of Jupyter notebooks into a cell-level representation.

Cells are preserved individually rather than grouped. Each cell carries the
heading path it sits under, so that grouping decisions belong to chunking
rather than to parsing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import nbformat

#: Output mimetypes carried through to the parsed representation. Everything
#: else (rendered HTML, widget state, Colab dataframe payloads) duplicates or
#: decorates content that is already available as plain text.
KEPT_MIMETYPES: tuple[str, ...] = ("text/plain",)

#: Longest output text kept for a single output before truncation.
MAX_OUTPUT_CHARS = 2000

#: Headings that mark pedagogical scaffolding rather than study material.
#: Matched case-insensitively against the heading that introduces a cell.
#: Derived from the current corpus and expected to grow.
PEDAGOGICAL_MARKERS: tuple[tuple[str, str], ...] = (
    ("instructor-only answers", "instructor-answers"),
    ("concept check", "concept-check"),
    ("thought experiment", "thought-experiment"),
)

_HEADING = re.compile(r"^(#{1,6})[ \t]+(\S.*?)[ \t]*$", re.MULTILINE)


@dataclass(frozen=True)
class Output:
    """A single execution output kept from a code cell."""

    output_type: str
    text: str
    truncated: bool


@dataclass(frozen=True)
class Cell:
    """One notebook cell with its position in the heading hierarchy."""

    index: int
    cell_type: str
    source: str
    heading_path: tuple[str, ...]
    tags: tuple[str, ...]
    outputs: tuple[Output, ...]
    dropped_outputs: tuple[str, ...]
    execution_count: int | None


@dataclass(frozen=True)
class ParsedNotebook:
    """A notebook reduced to its cells, headings, and useful outputs."""

    path: Path
    cells: tuple[Cell, ...]
    kernel_name: str | None
    language: str | None


def parse_notebook(path: Path) -> ParsedNotebook:
    """Read a notebook and return its cells with heading context.

    Raises FileNotFoundError if the path does not exist, and nbformat's
    ValidationError if the file is not a readable notebook.
    """
    if not path.is_file():
        raise FileNotFoundError(f"notebook not found: {path}")

    # nbformat ships no type information for read().
    notebook = nbformat.read(str(path), as_version=4)  # type: ignore[no-untyped-call]
    metadata = notebook.get("metadata", {})

    stack: list[tuple[int, str]] = []
    cells: list[Cell] = []

    for index, raw in enumerate(notebook.get("cells", [])):
        cell_type = str(raw.get("cell_type", ""))
        source = _as_text(raw.get("source", ""))

        heading_path, own_heading = _advance_headings(stack, cell_type, source)
        outputs, dropped = _collect_outputs(raw)

        cells.append(
            Cell(
                index=index,
                cell_type=cell_type,
                source=source,
                heading_path=heading_path,
                tags=_tags_for(own_heading),
                outputs=outputs,
                dropped_outputs=dropped,
                execution_count=raw.get("execution_count"),
            )
        )

    return ParsedNotebook(
        path=path,
        cells=tuple(cells),
        kernel_name=metadata.get("kernelspec", {}).get("name"),
        language=metadata.get("language_info", {}).get("name"),
    )


def _as_text(source: object) -> str:
    """Normalise a cell source, which may be a string or a list of lines."""
    if isinstance(source, list):
        return "".join(str(line) for line in source)
    return str(source)


def _advance_headings(
    stack: list[tuple[int, str]], cell_type: str, source: str
) -> tuple[tuple[str, ...], str | None]:
    """Update the heading stack for a cell and return its path.

    A cell's path includes its own first heading, so that the cell introducing
    a section is filed under that section rather than under its parent. Any
    further headings in the same cell apply to the cells that follow.
    """
    if cell_type != "markdown":
        return tuple(title for _, title in stack), None

    headings = [(len(hashes), title) for hashes, title in _HEADING.findall(source)]
    if not headings:
        return tuple(title for _, title in stack), None

    first_level, first_title = headings[0]
    _push_heading(stack, first_level, first_title)
    path = tuple(title for _, title in stack)

    for level, title in headings[1:]:
        _push_heading(stack, level, title)

    return path, first_title


def _push_heading(stack: list[tuple[int, str]], level: int, title: str) -> None:
    """Enter a heading, closing any open headings at the same or deeper level."""
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, title))


def _tags_for(heading: str | None) -> tuple[str, ...]:
    """Tag a cell whose heading marks it as pedagogical scaffolding."""
    if heading is None:
        return ()
    lowered = heading.lower()
    return tuple(tag for marker, tag in PEDAGOGICAL_MARKERS if marker in lowered)


def _collect_outputs(raw: object) -> tuple[tuple[Output, ...], tuple[str, ...]]:
    """Keep textual outputs and record the identity of everything discarded."""
    if not isinstance(raw, dict):
        return (), ()

    kept: list[Output] = []
    dropped: list[str] = []

    for output in raw.get("outputs", []):
        output_type = str(output.get("output_type", ""))

        if output_type == "stream":
            kept.append(_make_output(output_type, _as_text(output.get("text", ""))))
        elif output_type == "error":
            kept.append(_make_output(output_type, _error_text(output)))
        else:
            data = output.get("data", {})
            text = next((_as_text(data[m]) for m in KEPT_MIMETYPES if m in data), None)
            if text is not None:
                kept.append(_make_output(output_type, text))
            dropped.extend(
                f"{output_type}:{mimetype}"
                for mimetype in data
                if mimetype not in KEPT_MIMETYPES
            )

    return tuple(kept), tuple(dropped)


def _make_output(output_type: str, text: str) -> Output:
    """Build an output, truncating text that exceeds the size limit."""
    truncated = len(text) > MAX_OUTPUT_CHARS
    return Output(
        output_type=output_type,
        text=text[:MAX_OUTPUT_CHARS] if truncated else text,
        truncated=truncated,
    )


def _error_text(output: dict[str, object]) -> str:
    """Render an error output as its exception name and message."""
    name = str(output.get("ename", "")).strip()
    value = str(output.get("evalue", "")).strip()
    return f"{name}: {value}" if name and value else name or value

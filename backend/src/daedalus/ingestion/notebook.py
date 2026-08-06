"""
Jupyter notebook parsing.

Markdown cells and code sources are indexed. Cell outputs are stripped
apart from short text results and error messages: in the evaluation corpus
outputs account for 512,105 characters — 44% of raw volume — and consist
almost entirely of tensor dumps, progress bars, and warnings that match no
question a student would ask. See ADR-009.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import nbformat

from daedalus.config import constants
from daedalus.ingestion.types import ParsedDocument, Segment

__all__ = ["parse_notebook"]


logger = logging.getLogger(__name__)


_CELL_SEPARATOR = "\n\n"

# Outputs longer than this are almost always dumps rather than results.
_MAX_OUTPUT_CHARS = 500


def _join(source: Any) -> str:
    """Notebook sources are either a string or a list of lines."""

    if isinstance(source, list):
        return "".join(str(line) for line in source)

    return str(source)


def _useful_output(cell: Any) -> str:
    """
    Return the part of a cell's output worth indexing.

    Short text results carry meaning — a printed accuracy, a shape, a
    returned value. Errors carry meaning too, since notes often explain why
    something failed. Everything else is noise.
    """

    kept: list[str] = []

    for output in cell.get("outputs", []):
        kind = output.get("output_type")

        if kind == "error":
            name = output.get("ename", "Error")
            value = output.get("evalue", "")
            kept.append(f"{name}: {value}".strip())

        elif kind in {"stream", "execute_result", "display_data"}:
            if kind == "stream":
                text = _join(output.get("text", ""))
            else:
                text = _join(output.get("data", {}).get("text/plain", ""))

            text = text.strip()
            if text and len(text) <= _MAX_OUTPUT_CHARS:
                kept.append(text)

    return "\n".join(kept)


def parse_notebook(path: Path, doc_id: str, source_type: str) -> ParsedDocument:
    """Parse a notebook into flat text with one segment per cell."""

    notebook = nbformat.read(path, as_version=4)

    parts: list[str] = []
    segments: list[Segment] = []
    cursor = 0

    for index, cell in enumerate(notebook.get("cells", []), start=1):
        cell_type = cell.get("cell_type")
        source = _join(cell.get("source", "")).strip()

        if cell_type == "markdown":
            text = source
        elif cell_type == "code":
            output = _useful_output(cell)
            text = f"{source}\n\n{output}".strip() if output else source
        else:
            continue

        if not text:
            continue

        start = cursor
        end = start + len(text)

        parts.append(text)
        segments.append(
            Segment(
                start=start,
                end=end,
                extraction=constants.EXTRACTION_NOTEBOOK,
                page=index,
            )
        )

        cursor = end + len(_CELL_SEPARATOR)

    return ParsedDocument(
        doc_id=doc_id,
        text=_CELL_SEPARATOR.join(parts),
        segments=tuple(segments),
        source_type=source_type,
        n_pages=None,
    )

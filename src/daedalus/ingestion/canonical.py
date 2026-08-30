"""Conversion of parsed notebooks into the canonical document representation.

The parser in `notebook.py` stays faithful to the notebook format. This module
translates that into the format-independent representation in `document.py`.

"""

from __future__ import annotations

import hashlib

from daedalus.document import Document, Segment, SegmentKind
from daedalus.ingestion.notebook import ParsedNotebook

#: Appended to a segment's text when the output it came from was truncated,
#: so the loss is visible downstream without a format-specific field.
TRUNCATION_MARKER = "\n[output truncated]"

#: Value recorded as `Document.source_format` for notebooks.
SOURCE_FORMAT = "notebook"

#: Cell types that carry study material. Raw cells are excluded: their content
#: is marked by the notebook author as not for rendering, and is typically
#: preamble or conversion directives rather than material to be studied.
HANDLED_CELL_TYPES = ("markdown", "code")


def stable_document_id(content: str) -> str:
    """Return a stable identifier derived from a document's content.

    The identifier must depend only on `content`, so that the same material
    ingested twice — including from a different path — produces the same id,
    and any change to the content produces a different one.

    Return the first 16 characters of the SHA-256 hex digest of the UTF-8
    encoded content.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def extract_title(parsed: ParsedNotebook) -> str | None:
    """Return the document's title, or None if it has no headings.

    The title is the outermost heading of the first cell that sits under any
    heading — that is, the first element of the first non-empty heading path.
    """
    for cell in parsed.cells:
        if cell.heading_path:
            return cell.heading_path[0]

    return None


def notebook_to_document(parsed: ParsedNotebook) -> Document:
    """Convert a parsed notebook into a canonical document.

    Rules:

    * Cells whose type is not in HANDLED_CELL_TYPES, and cells whose source is
      empty or whitespace-only, are skipped entirely and do not contribute to
      `doc_id`.
    * A markdown cell becomes one PROSE segment holding the cell source.
    * A code cell becomes one CODE segment holding the cell source, followed
      by one OUTPUT segment per kept output, in order.
    * Each OUTPUT segment's `parent_ordinal` is the ordinal of the CODE
      segment from the same cell.
    * An OUTPUT segment whose source output was truncated has
      TRUNCATION_MARKER appended to its text.
    * `ordinal` counts segments, not cells, and runs contiguously from 0.
    * `heading_path` and `tags` are copied unchanged from the cell onto every
      segment produced from it, outputs included.
    * `locator` is `"cell:{index}"`, using the cell's index in the notebook.
    * `doc_id` comes from stable_document_id over the joined cell sources,
      separated by newlines. `source_format` is SOURCE_FORMAT, `source_path`
      is the parsed notebook's path, and `title` comes from extract_title.
    """
    segments: list[Segment] = []
    source_parts: list[str] = []

    for cell in parsed.cells:
        if cell.cell_type not in HANDLED_CELL_TYPES or not cell.source.strip():
            continue

        source_parts.append(cell.source)

        locator = f"cell:{cell.index}"

        if cell.cell_type == "markdown":
            segments.append(
                Segment(
                    ordinal=len(segments),
                    kind=SegmentKind.PROSE,
                    text=cell.source,
                    heading_path=cell.heading_path,
                    tags=cell.tags,
                    locator=locator,
                )
            )

        elif cell.cell_type == "code":
            code_ordinal = len(segments)

            segments.append(
                Segment(
                    ordinal=code_ordinal,
                    kind=SegmentKind.CODE,
                    text=cell.source,
                    heading_path=cell.heading_path,
                    tags=cell.tags,
                    locator=locator,
                )
            )

            for output in cell.outputs:
                output_text = output.text

                if output.truncated:
                    output_text += TRUNCATION_MARKER

                segments.append(
                    Segment(
                        ordinal=len(segments),
                        kind=SegmentKind.OUTPUT,
                        text=output_text,
                        heading_path=cell.heading_path,
                        tags=cell.tags,
                        locator=locator,
                        parent_ordinal=code_ordinal,
                    )
                )

    document_content = "\n".join(source_parts)

    return Document(
        doc_id=stable_document_id(document_content),
        source_path=parsed.path,
        source_format=SOURCE_FORMAT,
        title=extract_title(parsed),
        segments=tuple(segments),
    )

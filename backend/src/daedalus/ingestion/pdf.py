"""
PDF parsing with per-page extraction routing.

Each page is probed for a usable text layer. Pages that have one are read
directly; pages that do not are sent to OCR. Routing per page rather than
per document handles the common case of a digital PDF containing a few
scanned or photographed pages. See ADR-009.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pymupdf
import pymupdf4llm

from daedalus.config import constants
from daedalus.core.exceptions import ExtractionError
from daedalus.ingestion.types import ParsedDocument, Segment

__all__ = ["parse_pdf"]


logger = logging.getLogger(__name__)


_PAGE_SEPARATOR = "\n\n"


def _ocr_page(path: Path, page_number: int) -> str:
    """
    Extract text from a page with no usable text layer.

    Not yet implemented. No document in the evaluation corpus reaches this
    branch — every PDF has a clean text layer — so it is left unbuilt
    rather than guessed at. Raising loudly is better than silently
    returning nothing and producing a document with missing pages.
    """

    raise ExtractionError(
        f"{path.name} page {page_number} has no text layer and OCR is not yet implemented"
    )


def parse_pdf(path: Path, doc_id: str, source_type: str) -> ParsedDocument:
    """
    Parse a PDF into flat text with one segment per page.

    Pages are joined by a blank line; each segment records the character
    range its page occupies in the returned text.
    """

    document = pymupdf.open(path)

    try:
        page_count = document.page_count

        # Probe every page's raw text layer before doing the more expensive
        # markdown conversion, so routing is decided on cheap evidence.
        text_layer_pages = [
            index
            for index in range(page_count)
            if len(document[index].get_text().strip()) >= constants.TEXT_LAYER_MIN_CHARS
        ]

        rendered: dict[int, str] = {}

        if text_layer_pages:
            pages = pymupdf4llm.to_markdown(
                document,
                pages=text_layer_pages,
                page_chunks=True,
                show_progress=False,
            )
            rendered = {
                index: page["text"] for index, page in zip(text_layer_pages, pages, strict=True)
            }

        parts: list[str] = []
        segments: list[Segment] = []
        cursor = 0

        for index in range(page_count):
            if index in rendered:
                text = rendered[index].strip()
                extraction = constants.EXTRACTION_TEXT
            else:
                logger.info("%s page %d has no text layer, routing to OCR", path.name, index + 1)
                text = _ocr_page(path, index + 1).strip()
                extraction = constants.EXTRACTION_OCR

            if not text:
                continue

            start = cursor
            end = start + len(text)

            parts.append(text)
            segments.append(Segment(start=start, end=end, extraction=extraction, page=index + 1))

            cursor = end + len(_PAGE_SEPARATOR)

        return ParsedDocument(
            doc_id=doc_id,
            text=_PAGE_SEPARATOR.join(parts),
            segments=tuple(segments),
            source_type=source_type,
            n_pages=page_count,
        )

    finally:
        document.close()

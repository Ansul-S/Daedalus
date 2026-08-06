"""
Dispatching a file to the right parser.

The routing rule is documented in ADR-009. This module owns the extension
mapping and the document identifier; the per-format decisions (such as
which PDF pages need OCR) belong to the parsers themselves.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

from daedalus.config import constants
from daedalus.core.exceptions import UnsupportedFileTypeError
from daedalus.ingestion.markdown import parse_markdown
from daedalus.ingestion.notebook import parse_notebook
from daedalus.ingestion.pdf import parse_pdf
from daedalus.ingestion.types import ParsedDocument

__all__ = ["make_doc_id", "parse"]


logger = logging.getLogger(__name__)


_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def make_doc_id(path: Path) -> str:
    """
    Derive a stable, readable identifier from a filename.

    Used as the primary key and as the filename of the frozen parsed text
    that evaluation labels anchor to, so it must be stable across runs and
    safe as a path component.
    """

    normalized = unicodedata.normalize("NFKD", path.stem)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    slug = _SLUG_STRIP.sub("-", ascii_only.lower()).strip("-")

    return slug or "document"


def parse(path: Path, *, source_type: str = "unknown", doc_id: str | None = None) -> ParsedDocument:
    """
    Parse any supported file into text with provenance.

    Raises ``UnsupportedFileTypeError`` for extensions the pipeline does not
    handle, and ``ExtractionError`` when a supported file cannot be read.
    """

    suffix = path.suffix.lower()
    identifier = doc_id or make_doc_id(path)

    if suffix not in constants.SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"{path.name}: '{suffix}' is not supported "
            f"(expected one of {sorted(constants.SUPPORTED_EXTENSIONS)})"
        )

    logger.info("Parsing %s as %s", path.name, suffix)

    if suffix == ".pdf":
        return parse_pdf(path, identifier, source_type)

    if suffix == ".ipynb":
        return parse_notebook(path, identifier, source_type)

    if suffix == ".md":
        return parse_markdown(path, identifier, source_type)

    # Images are the remaining supported case. Imported lazily so that the
    # vision dependencies are only touched when an image is actually seen.
    from daedalus.ingestion.image import parse_image

    return parse_image(path, identifier, source_type)

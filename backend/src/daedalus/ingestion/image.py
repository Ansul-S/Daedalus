"""
Image parsing via a vision-language model.

Standalone images in this corpus are diagrams, not scanned text — meaning
lives in spatial layout, arrows, mathematical notation, and tables. OCR
flattens all of that into an unordered bag of strings, so images route to
a VLM instead. See ADR-009.

Not yet implemented: it requires a running Ollama with the vision model
pulled. The prompt below is the contract that implementation must satisfy.
"""

from __future__ import annotations

from pathlib import Path

from daedalus.core.exceptions import ExtractionError
from daedalus.ingestion.types import ParsedDocument

__all__ = ["VISION_PROMPT", "parse_image"]


VISION_PROMPT = """\
Transcribe this image into structured Markdown for a study-notes search index.

Rules:
- Preserve headings and the reading order a human would follow.
- Render tables as Markdown tables.
- Render mathematical notation as LaTeX, including superscripts and subscripts.
- For diagrams and flowcharts, describe the topology in prose: state which
  box connects to which, and in what direction. The arrows carry the meaning.
- Transcribe only what is present. Do not explain, summarise, or add context.
"""


def parse_image(path: Path, doc_id: str, source_type: str) -> ParsedDocument:
    """
    Parse an image into text using the configured vision model.

    Raises until implemented. Note for the implementation: vision output is
    nondeterministic, so the parsed text for the evaluation corpus must be
    frozen and committed once produced, or every character offset in the
    evaluation labels shifts on re-parse.
    """

    raise ExtractionError(
        f"{path.name}: vision extraction is not yet implemented "
        "(requires Ollama running with the vision model pulled)"
    )

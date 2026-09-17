"""PDF parsing with Docling: layout, reading order, tables, and LaTeX for formulas."""

from functools import lru_cache
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from app.ingest.chunking import ParsedDocument
from app.ingest.docling_blocks import document_blocks, document_title


@lru_cache(maxsize=2)
def _converter(ocr: bool, formulas: bool) -> DocumentConverter:
    # Building a converter loads the layout and table models, so converters are reused.
    options = PdfPipelineOptions(do_ocr=ocr, do_formula_enrichment=formulas)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def parse_pdf(
    path: Path, *, fallback_title: str, ocr: bool = False, formulas: bool = True
) -> ParsedDocument:
    """`ocr` is needed for scanned pages; `formulas` turns equations into LaTeX (slower)."""
    result = _converter(ocr, formulas).convert(path)
    doc = result.document
    title = document_title(doc) or fallback_title
    blocks = document_blocks(doc, title)
    if not blocks:
        hint = "" if ocr else "; if the PDF is scanned, ingest it again with OCR"
        raise ValueError(f"no text found in the PDF{hint}")
    details = {
        "pages": doc.num_pages(),
        "ocr": ocr,
        "formulas": formulas,
        "conversion": result.status.value,
    }
    return ParsedDocument(title=title, blocks=blocks, locator="page", details=details)

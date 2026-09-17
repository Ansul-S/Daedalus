"""End-to-end PDF parsing with Docling's layout, table and formula models (a few seconds once
the models are cached). Run with `make test-slow`."""

from pathlib import Path

import pytest

pytest.importorskip("docling")
from app.ingest.pdf import parse_pdf  # noqa: E402

pytestmark = pytest.mark.slow

# (font, size, x, y, text) on a US Letter page; y counts from the bottom
Line = tuple[str, int, int, int, str]
FONTS = {"regular": "Helvetica", "bold": "Helvetica-Bold", "math": "Times-Italic"}

INTRODUCTION = (
    "Recurrent networks read a sequence one step at a time and keep a hidden state that "
    "summarizes everything seen so far."
)
HIDDEN_STATE = (
    "The hidden state is a vector that is updated at every time step from the previous state "
    "and the current input."
)
SHARED_WEIGHTS = (
    "The same weights are used at every step, so the network can process sequences of any length."
)
TRAINING = (
    "Backpropagation through time unrolls the network and adds up the gradients of all steps, "
    "which can vanish or explode on long sequences."
)
REFERENCES = [
    "[1] S. Hochreiter and J. Schmidhuber. Long short-term memory. 1997.",
    "[2] J. Elman. Finding structure in time. 1990.",
]


def paragraph(y: int, text: str, width: int = 85) -> list[Line]:
    lines = [""]
    for word in text.split():
        if len(lines[-1]) + len(word) >= width:
            lines.append("")
        lines[-1] = f"{lines[-1]} {word}".strip()
    return [("regular", 11, 72, y - 14 * index, line) for index, line in enumerate(lines)]


def escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_pdf(path: Path, pages: list[list[Line]]) -> None:
    """A minimal PDF with text in the standard fonts, so no PDF library is needed."""
    # Objects 1 and 2 are the catalog and the page tree; the fonts follow.
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b""]
    fonts = []
    for name, base_font in FONTS.items():
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /%s >>" % base_font.encode())
        fonts.append(b"/%s %d 0 R" % (name.encode(), len(objects)))
    page_ids = []
    for lines in pages:
        text = "\n".join(
            f"BT /{font} {size} Tf {x} {y} Td ({escape(line)}) Tj ET"
            for font, size, x, y, line in lines
        ).encode()
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(text), text))
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << %s >> >> /Contents %d 0 R >>"
            % (b" ".join(fonts), len(objects))
        )
        page_ids.append(len(objects))
    kids = b" ".join(b"%d 0 R" % number for number in page_ids)
    objects[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, len(page_ids))

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref = len(pdf)
    pdf += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    pdf += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    pdf += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    path.write_bytes(bytes(pdf))


def test_a_pdf_becomes_blocks_with_sections_pages_and_latex(tmp_path) -> None:
    first_page = [
        ("bold", 20, 72, 720, "Recurrent Networks in Brief"),
        ("bold", 14, 72, 670, "1 Introduction"),
        *paragraph(645, INTRODUCTION),
        ("bold", 12, 72, 580, "1.1 Hidden State"),
        *paragraph(555, HIDDEN_STATE),
        # A displayed equation with its number at the right margin
        ("math", 16, 210, 500, "h = tanh(W x + U h + b)"),
        ("regular", 11, 520, 500, "(1)"),
        *paragraph(460, SHARED_WEIGHTS),
    ]
    second_page = [
        ("bold", 14, 72, 720, "2 Training"),
        *paragraph(695, TRAINING),
        ("bold", 14, 72, 620, "References"),
        *[("regular", 10, 72, 595 - 15 * i, entry) for i, entry in enumerate(REFERENCES)],
    ]
    path = tmp_path / "notes.pdf"
    write_pdf(path, [first_page, second_page])

    parsed = parse_pdf(path, fallback_title="notes")

    assert (parsed.title, parsed.locator) == ("Recurrent Networks in Brief", "page")
    assert (parsed.details["pages"], parsed.details["formulas"]) == (2, True)
    # Heading path of each block, and the page it starts on
    sections = {block.headings: block.start for block in parsed.blocks}
    assert sections == {
        ("1 Introduction",): 1,
        ("1 Introduction", "1.1 Hidden State"): 1,
        ("2 Training",): 2,
    }
    [formula] = [block for block in parsed.blocks if "formula" in block.content_types]
    assert formula.text.startswith("$$") and "tanh" in formula.text
    assert formula.headings == ("1 Introduction", "1.1 Hidden State")
    assert not any("Hochreiter" in block.text for block in parsed.blocks)

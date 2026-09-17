import pytest

pytest.importorskip("docling_core")
from docling_core.types.doc import (  # noqa: E402
    BoundingBox,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)

from app.ingest.docling_blocks import (  # noqa: E402
    compact_latex,
    document_blocks,
    document_title,
    join_small_caps,
)

TEXT = DocItemLabel.TEXT


def page(number: int) -> ProvenanceItem:
    return ProvenanceItem(page_no=number, bbox=BoundingBox(l=0, t=10, r=100, b=0), charspan=(0, 1))


def new_document(pages: int = 1) -> DoclingDocument:
    doc = DoclingDocument(name="test")
    for number in range(1, pages + 1):
        doc.add_page(page_no=number, size=Size(width=100, height=100))
    return doc


def table(rows: list[list[str]]) -> TableData:
    cells = [
        TableCell(
            text=text,
            start_row_offset_idx=row,
            end_row_offset_idx=row + 1,
            start_col_offset_idx=column,
            end_col_offset_idx=column + 1,
            column_header=row == 0,
        )
        for row, values in enumerate(rows)
        for column, text in enumerate(values)
    ]
    return TableData(num_rows=len(rows), num_cols=len(rows[0]), table_cells=cells)


def section_paths(headings: list[str]) -> list[tuple[str, ...]]:
    """Adds each heading at level 1, as PDF layout analysis does, with a paragraph below it,
    and returns the heading path of every paragraph that is kept."""
    doc = new_document()
    for heading in headings:
        doc.add_heading(heading, prov=page(1))
        doc.add_text(TEXT, "Some text.", prov=page(1))
    return [block.headings for block in document_blocks(doc, "Paper")]


def test_numbered_headings_are_nested() -> None:
    assert section_paths(["1 Introduction", "1.1  Background", "Key Insight", "2 Method"]) == [
        ("1 Introduction",),
        ("1 Introduction", "1.1 Background"),
        ("1 Introduction", "1.1 Background", "Key Insight"),
        ("2 Method",),
    ]


def test_ieee_headings_are_nested() -> None:
    headings = [
        "I. I NTRODUCTION",
        "A. Problem",
        # A subsection letter, not the numeral 100
        "C. Scope",
        # The paper repeats the number; a title in capitals still starts a section.
        "I. R ELATED W ORK",
        "B. Prior Work",
        "2) Graph Retrieval",
        "II. M ETHOD",
        "R EFERENCES",
    ]
    assert section_paths(headings) == [
        ("I. INTRODUCTION",),
        ("I. INTRODUCTION", "A. Problem"),
        ("I. INTRODUCTION", "C. Scope"),
        ("I. RELATED WORK",),
        ("I. RELATED WORK", "B. Prior Work"),
        ("I. RELATED WORK", "B. Prior Work", "2) Graph Retrieval"),
        ("II. METHOD",),
    ]


@pytest.mark.parametrize(
    "heading", ["References", "7 References", "VII. REFERENCES", "R EFERENCES", "Bibliography"]
)
def test_the_reference_list_is_dropped(heading: str) -> None:
    doc = new_document()
    doc.add_heading("6 Conclusion", prov=page(1))
    doc.add_text(TEXT, "Attention works.", prov=page(1))
    doc.add_heading(heading, prov=page(1))
    doc.add_text(TEXT, "[1] A. Author. A paper. 2017.", prov=page(1))

    assert [block.text for block in document_blocks(doc, "Paper")] == ["Attention works."]


def test_existing_heading_levels_are_kept() -> None:
    doc = new_document()
    doc.add_heading("Background", level=1)
    doc.add_heading("Attention", level=2)
    doc.add_text(TEXT, "Queries are matched against keys.")

    [block] = document_blocks(doc, "Paper")

    assert block.headings == ("Background", "Attention")
    assert (block.start, block.end) == (None, None)


def test_the_title_and_table_of_contents_are_left_out() -> None:
    doc = new_document()
    doc.add_heading("Attention Is All You Need", level=1)
    doc.add_table(table([["1 Background", "2"]]), label=DocItemLabel.DOCUMENT_INDEX)
    doc.add_heading("Background", level=2)
    doc.add_text(TEXT, "Recurrent models are sequential.")

    blocks = document_blocks(doc, "Attention  is all you need")

    assert [(block.text, block.headings) for block in blocks] == [
        ("Recurrent models are sequential.", ("Background",))
    ]


def test_blocks_keep_their_pages_and_content_types() -> None:
    doc = new_document(pages=3)
    doc.add_heading("1 Model", prov=page(1))
    spanning = doc.add_text(TEXT, "The fi xed weights are fi ne-tuned.", prov=page(1))
    spanning.prov.append(page(2))
    doc.add_text(TEXT, "Scores are divided by $\\sqrt{d_k}$.", prov=page(2))
    doc.add_formula("h _ { t } = \\tanh ( W _ { x } \\cdot x _ { t } )", prov=page(2))
    # Code keeps its text as it is.
    doc.add_code("fi le = open(path)", prov=page(2))
    doc.add_table(table([["Model", "BLEU"], ["Base", "27.3"]]), prov=page(3))

    blocks = document_blocks(doc, "Paper")

    assert [(b.text, sorted(b.content_types), b.start, b.end) for b in blocks] == [
        ("The fixed weights are fine-tuned.", ["text"], 1, 2),
        ("Scores are divided by $\\sqrt{d_k}$.", ["formula", "text"], 2, 2),
        ("$$h_{t} = \\tanh(W_{x} \\cdot x_{t})$$", ["formula"], 2, 2),
        ("```\nfi le = open(path)\n```", ["code"], 2, 2),
        ("| Model | BLEU |\n| - | - |\n| Base | 27.3 |", ["table"], 3, 3),
    ]


def test_blocks_link_to_the_innermost_heading_with_an_anchor() -> None:
    doc = new_document()
    doc.add_heading("3 Model Architecture", level=1)
    doc.add_heading("3.2 Attention", level=2)
    doc.add_heading("Scaled Dot-Product Attention", level=3)
    doc.add_text(TEXT, "Scores are scaled.")
    doc.add_heading("3.3 Feed-Forward  Networks", level=2)
    doc.add_text(TEXT, "Two linear layers.")
    anchors = {"3 Model Architecture": "S3", "3.2 Attention": "S3.SS2"}

    blocks = document_blocks(doc, "Paper", anchors)

    assert [block.anchor for block in blocks] == ["S3.SS2", "S3"]


def test_the_title_is_the_title_item_or_a_heading_that_opens_the_first_page() -> None:
    titled = new_document()
    titled.add_heading("Preface", prov=page(1))
    titled.add_title("RNN Intuition", prov=page(1))

    opened = new_document()
    opened.add_heading(" RNN Intuition ", prov=page(1))
    opened.add_text(TEXT, "Structured notes.", prov=page(1))

    untitled = new_document()
    untitled.add_text(TEXT, "Draft notes.", prov=page(1))
    untitled.add_heading("1 Introduction", prov=page(1))

    assert document_title(titled) == "RNN Intuition"
    assert document_title(opened) == "RNN Intuition"
    assert document_title(untitled) is None


@pytest.mark.parametrize(
    ("recognized", "expected"),
    [
        ("M e m o r y _ { t - 1 }", "Memory_{t - 1}"),
        ("\\frac { Q K ^ { T } } { \\sqrt { d _ { k } } }", "\\frac{QK^{T}}{\\sqrt{d_{k}}}"),
        # A space ends a control word, so it stays.
        ("W _ { h } \\cdot h _ { t }", "W_{h} \\cdot h_{t}"),
        ("x _ { 1 0 0 }", "x_{100}"),
        ("a + b", "a + b"),
    ],
)
def test_recognized_latex_is_compacted(recognized: str, expected: str) -> None:
    assert compact_latex(recognized) == expected


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("I NTRODUCTION", "INTRODUCTION"),
        ("III. I MPLEMENTATION", "III. IMPLEMENTATION"),
        ("IV. E XPERIMENTAL R ESULTS AND E VALUATION", "IV. EXPERIMENTAL RESULTS AND EVALUATION"),
        ("C OMPONENT C ONTRIBUTION A NALYSIS :", "COMPONENT CONTRIBUTION ANALYSIS :"),
        ("A PPENDIX A P ROOFS", "APPENDIX A PROOFS"),
        # Plain capitals: only one of three words looks split, so nothing is joined.
        ("APPENDIX A PROOFS", "APPENDIX A PROOFS"),
        ("B. Prior Work", "B. Prior Work"),
    ],
)
def test_small_capitals_are_joined(heading: str, expected: str) -> None:
    assert join_small_caps(heading) == expected

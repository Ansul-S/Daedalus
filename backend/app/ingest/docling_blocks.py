"""Turn a Docling document (parsed from a PDF or an HTML page) into blocks for the chunker."""

import re
from collections.abc import Iterable, Mapping

from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
    HierarchicalChunker,
)
from docling_core.transforms.serializer.markdown import MarkdownParams, MarkdownTableSerializer
from docling_core.types.doc import (
    ContentLayer,
    DocItem,
    DocItemLabel,
    DoclingDocument,
    FormulaItem,
    SectionHeaderItem,
    TitleItem,
)

from app.ingest.chunking import Block

# Heading numbering: "3.1 Title", IEEE-style "IV. TITLE", "B. Title" and "2) Title"
_ARABIC_NUMBER = re.compile(r"(\d+(?:\.\d+)*)\.?\s+\S")
_ROMAN_NUMBER = re.compile(r"([IVXLC]+)\.\s+(\S.*)")
_LETTER = re.compile(r"[A-Z]\.\s+\S")
_PARENTHESIZED_NUMBER = re.compile(r"\d+\)\s+\S")
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
_NUMBER_PREFIX = re.compile(r"\A(?:\d+(?:\.\d+)*\.?|[A-Z]+\.)\s+")
# Small capitals can come out of a PDF as "I NTRODUCTION": the large first letter is its own run.
_SMALL_CAPS_SPLIT = re.compile(r"\b([A-Z]) (?=[A-Z]{2,}\b)")
_CAPITALS_WORD = re.compile(r"\b[A-Z]{2,}\b")
_REFERENCE_TITLES = {"references", "bibliography", "referencesandnotes"}
_INLINE_MATH = re.compile(r"\$\$.+?\$\$|\$[^$\n]+\$", re.DOTALL)
# PDF text extraction sometimes splits a word after its opening ligature: "fi xed", "fi nal".
_SPLIT_LIGATURE = re.compile(r"(?<![\w-])(ffi|ffl|ff|fi|fl) (?=[a-z])")
_CONTENT_TYPES = {
    DocItemLabel.CODE: "code",
    DocItemLabel.FORMULA: "formula",
    DocItemLabel.TABLE: "table",
}

# LaTeX tokens: control words (\frac), control symbols (\,), whitespace, single characters
_LATEX_TOKEN = re.compile(r"\\[A-Za-z]+|\\.|\s+|.", re.DOTALL)
_CONTROL_WORD = re.compile(r"\\[A-Za-z]+")
_NO_SPACE_AFTER = {"_", "^", "{", "(", "["}
_NO_SPACE_BEFORE = {"_", "^", "{", "}", "(", ")", "[", "]", ","}


class _MarkdownSerializers(ChunkingSerializerProvider):
    """Docling's chunking serializer, but with compact Markdown tables instead of
    "row, column = value" sentences: easier to read in citations and for the LLM."""

    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            params=MarkdownParams(
                image_placeholder="",
                escape_underscores=False,
                escape_html=False,
                compact_tables=True,
            ),
        )


def document_blocks(
    doc: DoclingDocument, title: str, anchors: Mapping[str, str] | None = None
) -> list[Block]:
    """`anchors` maps heading text to a fragment identifier in the source page."""
    _prepare(doc)
    blocks: list[Block] = []
    for chunk in HierarchicalChunker(serializer_provider=_MarkdownSerializers()).chunk(doc):
        headings = tuple(chunk.meta.headings or ())
        if headings and _same_text(headings[0], title):
            headings = headings[1:]
        if any(_is_references(heading) for heading in headings):
            continue
        text = chunk.text.strip()
        if not text:
            continue
        items = chunk.meta.doc_items
        content_types = _content_types(items, text)
        if "code" not in content_types:
            text = _SPLIT_LIGATURE.sub(r"\1", text)
        pages = [prov.page_no for item in items for prov in item.prov]
        anchor = _anchor(headings, anchors) if anchors else None
        blocks.append(
            Block(
                text=text,
                headings=headings,
                content_types=content_types,
                start=min(pages, default=None),
                end=max(pages, default=None),
                anchor=anchor,
            )
        )
    return blocks


def document_title(doc: DoclingDocument) -> str | None:
    """The detected title, or a heading that opens the first page before any body text."""
    for item, _ in doc.iterate_items():
        if isinstance(item, TitleItem):
            return item.text.strip()
    for item, _ in doc.iterate_items(page_no=1):
        if isinstance(item, SectionHeaderItem):
            return item.text.strip()
        if isinstance(item, DocItem):
            break
    return None


def _prepare(doc: DoclingDocument) -> None:
    headers: list[SectionHeaderItem] = []
    for item, _ in doc.iterate_items():
        if not isinstance(item, DocItem):
            continue
        if item.label == DocItemLabel.DOCUMENT_INDEX:
            # A table of contents only repeats the headings.
            item.content_layer = ContentLayer.FURNITURE
        elif isinstance(item, FormulaItem) and item.text:
            item.text = compact_latex(item.text)
        elif isinstance(item, SectionHeaderItem):
            item.text = normalize_heading(item.text)
            headers.append(item)
    if len({header.level for header in headers}) == 1:
        _infer_heading_levels(headers)


def _infer_heading_levels(headers: Iterable[SectionHeaderItem]) -> None:
    """PDF layout analysis gives every heading the same level. Recover the outline from the
    numbering: "3" is level 1 and "3.1" level 2; in IEEE style "IV." is level 1, "B." level 2
    and "2)" level 3. An unnumbered heading sits one level below the last numbered one."""
    numbered_level = 0
    last_roman = 0
    for header in headers:
        header.text = join_small_caps(header.text)
        level = None
        if match := _ARABIC_NUMBER.match(header.text):
            level = match.group(1).count(".") + 1
        elif (match := _ROMAN_NUMBER.fullmatch(header.text)) and _is_section_numeral(
            match, last_roman
        ):
            last_roman = max(last_roman, _roman_value(match.group(1)))
            level = 1
        elif _LETTER.match(header.text):
            level = 2 if last_roman else 1
        elif _PARENTHESIZED_NUMBER.match(header.text) and last_roman:
            level = 3

        if level is None:
            header.level = numbered_level + 1
        else:
            numbered_level = header.level = level


def _is_section_numeral(match: re.Match[str], last_roman: int) -> bool:
    """Only the next numeral starts a section, so a subsection "C." is not read as 100. A title
    in capitals, as IEEE section titles are, may also repeat an earlier number: some papers
    have two sections "I."."""
    value = _roman_value(match.group(1))
    return value == last_roman + 1 or (value <= last_roman and match.group(2).isupper())


def join_small_caps(text: str) -> str:
    """Rejoin words split by small capitals ("R ELATED W ORK"). Only done when most words in
    capitals are split, so a heading in plain capitals such as "APPENDIX A PROOFS" is kept."""
    title = _NUMBER_PREFIX.sub("", text, count=1)
    splits = len(_SMALL_CAPS_SPLIT.findall(title))
    unsplit = len(_CAPITALS_WORD.findall(title)) - splits
    return _SMALL_CAPS_SPLIT.sub(r"\1", text) if splits > unsplit else text


def _roman_value(numeral: str) -> int:
    values = [_ROMAN_VALUES[symbol] for symbol in numeral]
    return sum(
        -value if index + 1 < len(values) and values[index + 1] > value else value
        for index, value in enumerate(values)
    )


def _is_references(heading: str) -> bool:
    title = _NUMBER_PREFIX.sub("", heading, count=1)
    return "".join(title.split()).casefold() in _REFERENCE_TITLES


def _content_types(items: Iterable[DocItem], text: str) -> frozenset[str]:
    kinds = {_CONTENT_TYPES.get(item.label, "text") for item in items}
    if "formula" not in kinds and _INLINE_MATH.search(text):
        kinds.add("formula")
    return frozenset(kinds)


def normalize_heading(text: str) -> str:
    return " ".join(text.split())


def _anchor(headings: tuple[str, ...], anchors: Mapping[str, str]) -> str | None:
    """The fragment of the innermost heading that has one."""
    for heading in reversed(headings):
        if anchor := anchors.get(normalize_heading(heading)):
            return anchor
    return None


def _same_text(a: str, b: str) -> bool:
    return normalize_heading(a).casefold() == normalize_heading(b).casefold()


def compact_latex(latex: str) -> str:
    """Remove the spaces a formula recognizer puts between tokens, so "M e m o r y _ { t }"
    becomes "Memory_{t}". Spaces are ignored in math mode, except after a control word such as
    "\\cdot x", so those are kept."""
    tokens = _LATEX_TOKEN.findall(latex.strip())
    kept: list[str] = []
    for index, token in enumerate(tokens):
        if not token.isspace():
            kept.append(token)
            continue
        before = kept[-1] if kept else ""
        after = tokens[index + 1] if index + 1 < len(tokens) else ""
        ends_command = bool(_CONTROL_WORD.fullmatch(before)) and after[:1].isalpha()
        if ends_command or not _is_spurious_space(before, after):
            kept.append(" ")
    return "".join(kept)


def _is_spurious_space(before: str, after: str) -> bool:
    return (
        before in _NO_SPACE_AFTER
        or after in _NO_SPACE_BEFORE
        # Letters or digits of one word or number: "M e m o r y", "1 0 0"
        or (len(before) == len(after) == 1 and before.isalnum() and after.isalnum())
    )

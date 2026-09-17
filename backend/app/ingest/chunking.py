"""Pack parsed blocks into chunks of roughly `min_tokens`–`max_tokens` tokens.

Every parser (PDF, arXiv HTML, notebook) produces a flat list of blocks: a paragraph, list,
table, formula or code cell, together with the heading path it sits under and where it came
from (page or cell). Packing then works the same way for every source:

- blocks are added to the current chunk until the next one would push it past `max_tokens`;
- a new section or subsection (a change in the first two heading levels) starts a new chunk
  once the current one has `min_tokens`, so small sections are merged instead of becoming
  fragments, and a chunk rarely mixes more than a few subsections;
- when the heading path changes inside a chunk, the new headings are written into the text as
  Markdown headings, so a chunk that spans sections keeps its structure;
- a chunk's section label names every section it covers, not just the one it starts in;
- a block longer than `max_tokens` is split on paragraph, line and word boundaries; code
  blocks keep their fences and tables repeat their header row.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

TokenCounter = Callable[[str], int]

CONTENT_TYPE_ORDER = ("text", "code", "formula", "table")
SECTION_SEPARATOR = " > "
# Between the sections a chunk covers: "3 Model > 3.3 Feed-Forward Networks · 3.4 Embeddings"
SECTION_LIST_SEPARATOR = " · "
# Heading levels whose change starts a new chunk: sections and subsections
SECTION_DEPTH = 2

_FENCED = re.compile(r"((`{3,}|~{3,})[^\n]*\n)(.*)(\n\2)", re.DOTALL)


@dataclass(frozen=True)
class Block:
    text: str
    headings: tuple[str, ...] = ()
    content_types: frozenset[str] = frozenset({"text"})
    # First and last page (PDF) or cell (notebook), 1-based
    start: int | None = None
    end: int | None = None
    # Fragment identifier of the section in an HTML source
    anchor: str | None = None


@dataclass
class ParsedDocument:
    """What a parser returns."""

    title: str
    blocks: list[Block]
    # What Block.start / Block.end count
    locator: Literal["page", "cell"] | None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkDraft:
    text: str
    # Heading paths of the chunk's blocks, in order, without repeats
    sections: tuple[tuple[str, ...], ...]
    content_types: list[str]
    token_count: int
    start: int | None
    end: int | None
    anchor: str | None

    @property
    def section(self) -> str | None:
        """The sections the chunk covers, named below their common parent:
        "3 Model Architecture > 3.3 Feed-Forward Networks · 3.4 Embeddings"."""
        paths = [path for path in self.sections if path]
        if not paths:
            return None
        parent = paths[0]
        for path in paths[1:]:
            parent = parent[: _common_length(parent, path)]
        children = dict.fromkeys(path[len(parent)] for path in paths if len(path) > len(parent))
        if children:
            return SECTION_SEPARATOR.join([*parent, SECTION_LIST_SEPARATOR.join(children)])
        return SECTION_SEPARATOR.join(parent)


def pack_blocks(
    blocks: Sequence[Block], count_tokens: TokenCounter, max_tokens: int, min_tokens: int
) -> list[ChunkDraft]:
    chunks: list[ChunkDraft] = []
    current: list[Block] = []
    current_tokens = 0

    for block in blocks:
        for piece in split_block(block, count_tokens, max_tokens):
            if current:
                candidate_tokens = count_tokens(render([*current, piece]))
                new_section = piece.headings[:SECTION_DEPTH] != current[-1].headings[:SECTION_DEPTH]
                if candidate_tokens > max_tokens or (new_section and current_tokens >= min_tokens):
                    chunks.append(_merge(current, current_tokens))
                    current = []
            current.append(piece)
            current_tokens = count_tokens(render(current))

    if current:
        chunks.append(_merge(current, current_tokens))
    return chunks


def render(blocks: Sequence[Block]) -> str:
    """Join block texts, writing a heading line wherever the heading path changes."""
    parts: list[str] = []
    previous = blocks[0].headings
    for block in blocks:
        if block.headings != previous:
            parts.extend(_heading_lines(previous, block.headings))
            previous = block.headings
        parts.append(block.text)
    return "\n\n".join(parts)


def _heading_lines(previous: tuple[str, ...], new: tuple[str, ...]) -> list[str]:
    common = _common_length(previous, new)
    # Moving back up to a parent section: repeat the parent's heading so the text says so.
    first = min(common, len(new) - 1) if new else 0
    return [f"{'#' * min(depth + 1, 6)} {new[depth]}" for depth in range(first, len(new))]


def _common_length(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    """How many leading headings two heading paths share."""
    length = 0
    while length < min(len(a), len(b)) and a[length] == b[length]:
        length += 1
    return length


def _merge(blocks: list[Block], token_count: int) -> ChunkDraft:
    starts = [block.start for block in blocks if block.start is not None]
    ends = [block.end for block in blocks if block.end is not None]
    content_types = set().union(*(block.content_types for block in blocks))
    return ChunkDraft(
        text=render(blocks),
        sections=tuple(dict.fromkeys(block.headings for block in blocks)),
        content_types=[kind for kind in CONTENT_TYPE_ORDER if kind in content_types],
        token_count=token_count,
        start=min(starts, default=None),
        end=max(ends, default=None),
        anchor=next((block.anchor for block in blocks if block.anchor), None),
    )


def split_block(block: Block, count_tokens: TokenCounter, max_tokens: int) -> list[Block]:
    if count_tokens(block.text) <= max_tokens:
        return [block]

    if fenced := _FENCED.fullmatch(block.text):
        opening, _, body, closing = fenced.groups()
        budget = max_tokens - count_tokens(opening + closing)
        pieces = [opening + piece + closing for piece in split_text(body, count_tokens, budget)]
    elif _is_table(block.text):
        # Markdown table: the first two lines are the header row and the separator.
        title_row, separator_row, rows = block.text.split("\n", 2)
        header = f"{title_row}\n{separator_row}"
        budget = max_tokens - count_tokens(header) - 1
        pieces = [f"{header}\n{piece}" for piece in split_text(rows, count_tokens, budget, ("\n",))]
    else:
        pieces = split_text(block.text, count_tokens, max_tokens)
    return [replace(block, text=piece) for piece in pieces]


def _is_table(text: str) -> bool:
    lines = text.split("\n")
    return len(lines) > 2 and all(line.startswith("|") for line in lines)


def split_text(
    text: str,
    count_tokens: TokenCounter,
    budget: int,
    separators: tuple[str, ...] = ("\n\n", "\n", " "),
) -> list[str]:
    """Split text into pieces of at most `budget` tokens, cutting at the coarsest separator
    that works: paragraphs, then lines, then words."""
    if count_tokens(text) <= budget:
        return [text]
    if not separators:
        return _split_characters(text, count_tokens, budget)

    separator, finer = separators[0], separators[1:]
    pieces: list[str] = []
    current = ""
    for part in text.split(separator):
        candidate = f"{current}{separator}{part}" if current else part
        if count_tokens(candidate) <= budget:
            current = candidate
            continue
        if current:
            pieces.append(current)
        if count_tokens(part) <= budget:
            current = part
        else:
            *complete, current = split_text(part, count_tokens, budget, finer)
            pieces.extend(complete)
    if current:
        pieces.append(current)
    return pieces


def _split_characters(text: str, count_tokens: TokenCounter, budget: int) -> list[str]:
    """Last resort for a single "word" longer than the budget, such as an encoded blob."""
    pieces: list[str] = []
    while text:
        size = min(len(text), budget)
        while size > 1 and count_tokens(text[:size]) > budget:
            size //= 2
        pieces.append(text[:size])
        text = text[size:]
    return pieces

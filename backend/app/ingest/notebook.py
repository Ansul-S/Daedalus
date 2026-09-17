"""Jupyter notebooks: Markdown cells become text, code cells become fenced code blocks.

Short text outputs (printed results, small tables) are kept below their code cell because
they often hold the numbers the notebook is about. Images, HTML, widgets, errors, warnings
and the tail of long outputs are dropped.
"""

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import nbformat

from app.ingest.chunking import Block, ParsedDocument

MAX_OUTPUT_CHARS = 1000

_HEADING = re.compile(r" {0,3}(#{1,6})\s+(.+?)(?:\s+#+)?\s*")
_FENCE = re.compile(r" {0,3}(```|~~~)")
_EMBEDDED_IMAGE = re.compile(
    r"!\[[^\]]*\]\((?:data:|attachment:)[^)]*\)|<img\b[^>]*>", re.IGNORECASE
)
_HTML_TAG = re.compile(r"<[^>]+>")
_INLINE_MATH = re.compile(r"\$\$.+?\$\$|\$[^$\n]+\$", re.DOTALL)
# `!pip install ...`, `%matplotlib inline`, `%%capture`
_SHELL_OR_MAGIC = re.compile(r"\s*[!%]")
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# Placeholder text of rich outputs, e.g. "<IPython.core.display.HTML object>"
_OBJECT_REPR = re.compile(r"<[\w.]+ (?:object|at)\b[^>]*>")
_BACKTICKS = re.compile(r"`+")


def parse_notebook(path: Path, *, fallback_title: str) -> ParsedDocument:
    notebook = nbformat.read(path, as_version=4)
    language = _language(notebook.metadata)
    title: str | None = None
    outline: list[tuple[int, str]] = []  # (level, heading) of the enclosing sections
    blocks: list[Block] = []

    def headings() -> tuple[str, ...]:
        path = [heading for _, heading in outline]
        return tuple(path[1:] if path and path[0] == title else path)

    for number, cell in enumerate(notebook.cells, start=1):
        if cell.cell_type == "markdown":
            for level, text in _markdown_parts(cell.source):
                if level:
                    outline = [entry for entry in outline if entry[0] < level] + [(level, text)]
                    if level == 1 and title is None:
                        title = text
                else:
                    kinds = {"text", "formula"} if _INLINE_MATH.search(text) else {"text"}
                    blocks.append(Block(text, headings(), frozenset(kinds), number, number))
        elif cell.cell_type == "code" and (code := _code(cell.source)):
            code_block = Block(
                _fenced(language, code), headings(), frozenset({"code"}), number, number
            )
            blocks.append(code_block)
            if output := _text_output(cell.get("outputs", [])):
                blocks.append(
                    Block(
                        _fenced("output", output), headings(), frozenset({"code"}), number, number
                    )
                )

    details = {"cells": len(notebook.cells), "language": language}
    return ParsedDocument(
        title=title or fallback_title, blocks=blocks, locator="cell", details=details
    )


def _language(metadata: Any) -> str:
    language_info = metadata.get("language_info") or {}
    kernelspec = metadata.get("kernelspec") or {}
    return language_info.get("name") or kernelspec.get("language") or "python"


def _markdown_parts(source: str) -> Iterator[tuple[int, str]]:
    """Yield (level, heading) for each heading and (0, text) for the text between headings."""
    lines: list[str] = []
    in_fence = False
    for line in _EMBEDDED_IMAGE.sub("", source).splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        heading = None if in_fence else _HEADING.fullmatch(line)
        if heading is None:
            lines.append(line)
            continue
        if text := "\n".join(lines).strip():
            yield 0, text
        lines = []
        if heading_text := _HTML_TAG.sub("", heading.group(2)).replace("**", "").strip():
            yield len(heading.group(1)), heading_text
    if text := "\n".join(lines).strip():
        yield 0, text


def _code(source: str) -> str:
    lines = [line.rstrip() for line in source.splitlines() if not _SHELL_OR_MAGIC.match(line)]
    return "\n".join(lines).strip("\n")


def _text_output(outputs: list[Any]) -> str:
    parts: list[str] = []
    for output in outputs:
        kind = output.get("output_type")
        if kind == "stream":
            if output.get("name") == "stdout":
                parts.append(output.get("text", ""))
        elif kind in ("execute_result", "display_data"):
            data = output.get("data", {})
            if any(mime.startswith("image/") or "widget" in mime for mime in data):
                continue
            text = data.get("text/plain", "")
            if not _OBJECT_REPR.fullmatch(text.strip()):
                parts.append(text)
    text = _ANSI_ESCAPE.sub("", "\n".join(part.rstrip() for part in parts)).strip()
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    cut = text[:MAX_OUTPUT_CHARS]
    if "\n" in cut:
        cut = cut.rsplit("\n", 1)[0]
    return f"{cut}\n... (output truncated)"


def _fenced(info: str, body: str) -> str:
    # The fence must be longer than any backtick run inside the body.
    longest = max((len(run) for run in _BACKTICKS.findall(body)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{info}\n{body}\n{fence}"

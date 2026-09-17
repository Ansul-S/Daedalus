from app.ingest.chunking import Block, pack_blocks, render, split_block, split_text


def words(text: str) -> int:
    return len(text.split())


def block(text: str, *headings: str, kinds: tuple[str, ...] = ("text",), page: int | None = None):
    return Block(text, headings, frozenset(kinds), page, page)


def test_a_new_section_starts_a_chunk_only_after_the_minimum() -> None:
    blocks = [
        block("a " * 10, "1 Intro"),
        block("b " * 10, "2 Method"),
        block("c " * 10, "3 Results"),
    ]

    chunks = pack_blocks(blocks, words, max_tokens=100, min_tokens=20)

    # "1 Intro" alone is below the minimum, so "2 Method" joins it; "3 Results" does not.
    assert [chunk.section for chunk in chunks] == ["1 Intro", "3 Results"]
    assert "# 2 Method" in chunks[0].text
    assert chunks[0].token_count == words(chunks[0].text)


def test_chunks_never_exceed_the_maximum() -> None:
    blocks = [block("word " * 30, "Section") for _ in range(5)]

    chunks = pack_blocks(blocks, words, max_tokens=70, min_tokens=10)

    assert [chunk.token_count for chunk in chunks] == [60, 60, 30]


def test_heading_changes_inside_a_chunk_are_written_as_markdown() -> None:
    text = render([block("one", "A"), block("two", "A", "B"), block("three", "A")])
    assert text == "one\n\n## B\n\ntwo\n\n# A\n\nthree"


def test_locations_and_content_types_are_combined() -> None:
    blocks = [
        block("text", "S", page=3),
        block("| a |\n| - |\n| 1 |", "S", kinds=("table",), page=4),
    ]

    [chunk] = pack_blocks(blocks, words, max_tokens=100, min_tokens=10)

    assert (chunk.start, chunk.end) == (3, 4)
    assert chunk.content_types == ["text", "table"]


def test_long_text_is_split_at_paragraphs_first() -> None:
    text = "\n\n".join(" ".join([f"p{n}"] * 8) for n in range(3))
    assert split_text(text, words, budget=10) == [
        " ".join(["p0"] * 8),
        " ".join(["p1"] * 8),
        " ".join(["p2"] * 8),
    ]


def test_long_code_keeps_its_fences() -> None:
    code = "\n".join(f"x{n} = {n}" for n in range(30))
    pieces = split_block(block(f"```python\n{code}\n```", kinds=("code",)), words, max_tokens=20)

    assert len(pieces) > 1
    for piece in pieces:
        assert piece.text.startswith("```python\n") and piece.text.endswith("\n```")
        assert words(piece.text) <= 20
    assert "\n".join(piece.text[10:-4] for piece in pieces) == code


def test_long_output_with_a_longer_fence_is_split_correctly() -> None:
    body = "\n".join(f"line {n} ```" for n in range(20))
    pieces = split_block(block(f"````output\n{body}\n````", kinds=("code",)), words, 15)
    assert all(p.text.startswith("````output\n") and p.text.endswith("\n````") for p in pieces)


def test_long_tables_repeat_their_header() -> None:
    rows = "\n".join(f"| model {n} | {n} |" for n in range(20))
    table = f"| name | score |\n| - | - |\n{rows}"

    pieces = split_block(block(table, kinds=("table",)), words, max_tokens=30)

    assert len(pieces) > 1
    assert all(piece.text.startswith("| name | score |\n| - | - |\n") for piece in pieces)
    assert sum(piece.text.count("| model") for piece in pieces) == 20


def test_an_unbreakable_word_is_cut_by_characters() -> None:
    pieces = split_text("x" * 50, len, budget=20)
    assert "".join(pieces) == "x" * 50
    assert max(len(piece) for piece in pieces) <= 20

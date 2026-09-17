import pytest

nbformat = pytest.importorskip("nbformat")
from nbformat.v4 import (  # noqa: E402
    new_code_cell,
    new_markdown_cell,
    new_notebook,
    new_output,
)

from app.ingest.notebook import MAX_OUTPUT_CHARS, parse_notebook  # noqa: E402


def write_notebook(tmp_path, cells):
    notebook = new_notebook(cells=cells)
    notebook.metadata["language_info"] = {"name": "python"}
    path = tmp_path / "lesson.ipynb"
    nbformat.write(notebook, path)
    return path


def test_markdown_headings_become_the_section_path(tmp_path) -> None:
    path = write_notebook(
        tmp_path,
        [
            new_markdown_cell("# Retrieval QnA\n\nIntro text."),
            new_markdown_cell("## Retriever\nFAISS finds chunks.\n### Scores\nCosine $s(q, d)$."),
            new_markdown_cell("## Reader\nBERT extracts the span."),
        ],
    )

    parsed = parse_notebook(path, fallback_title="lesson")

    assert parsed.title == "Retrieval QnA"
    assert parsed.locator == "cell"
    assert [(block.text, block.headings, block.start) for block in parsed.blocks] == [
        ("Intro text.", (), 1),
        ("FAISS finds chunks.", ("Retriever",), 2),
        ("Cosine $s(q, d)$.", ("Retriever", "Scores"), 2),
        ("BERT extracts the span.", ("Reader",), 3),
    ]
    assert parsed.blocks[2].content_types == {"text", "formula"}


def test_code_cells_become_fenced_blocks_without_shell_lines(tmp_path) -> None:
    code = (
        "!pip install faiss-cpu\n%matplotlib inline\nimport faiss\nindex = faiss.IndexFlatIP(384)"
    )
    path = write_notebook(tmp_path, [new_markdown_cell("# Title"), new_code_cell(code)])

    [block] = parse_notebook(path, fallback_title="lesson").blocks

    assert block.text == "```python\nimport faiss\nindex = faiss.IndexFlatIP(384)\n```"
    assert (block.start, block.content_types) == (2, {"code"})


def test_short_text_outputs_are_kept_and_the_rest_dropped(tmp_path) -> None:
    outputs = [
        new_output("stream", name="stdout", text="Time taken: 0.32 seconds\n"),
        new_output("stream", name="stderr", text="Warning: deprecated\n"),
        new_output("execute_result", data={"text/plain": "retriever  time\n0  FAISS  0.038"}),
        new_output("display_data", data={"image/png": "iVBOR", "text/plain": "<Figure>"}),
        new_output("display_data", data={"text/plain": "<IPython.core.display.HTML object>"}),
        new_output("error", ename="ValueError", evalue="bad", traceback=[]),
    ]
    path = write_notebook(tmp_path, [new_code_cell("run()", outputs=outputs)])

    code, output = parse_notebook(path, fallback_title="lesson").blocks

    assert code.text == "```python\nrun()\n```"
    assert output.text == (
        "```output\nTime taken: 0.32 seconds\nretriever  time\n0  FAISS  0.038\n```"
    )


def test_long_outputs_are_truncated_and_fences_stay_valid(tmp_path) -> None:
    printed = "\n".join(f"row {n} ```" for n in range(500))
    cell = new_code_cell("show()", outputs=[new_output("stream", name="stdout", text=printed)])
    path = write_notebook(tmp_path, [cell])

    output = parse_notebook(path, fallback_title="lesson").blocks[1].text

    assert output.startswith("````output\nrow 0 ```")
    assert output.endswith("... (output truncated)\n````")
    assert len(output) < MAX_OUTPUT_CHARS + 50


def test_embedded_images_are_removed_and_the_file_name_is_the_fallback_title(tmp_path) -> None:
    image = "![plot](data:image/png;base64," + "A" * 5000 + ")"
    path = write_notebook(tmp_path, [new_markdown_cell(f"Before {image} after")])

    parsed = parse_notebook(path, fallback_title="lesson")

    assert parsed.title == "lesson"
    assert [block.text for block in parsed.blocks] == ["Before  after"]


def test_headings_inside_code_fences_are_ignored(tmp_path) -> None:
    markdown = "## Setup\n```bash\n# not a heading\n```"
    path = write_notebook(tmp_path, [new_markdown_cell(markdown)])

    [block] = parse_notebook(path, fallback_title="lesson").blocks

    assert block.headings == ("Setup",)
    assert "# not a heading" in block.text

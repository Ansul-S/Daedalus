"""
Freezing and reloading the parsed corpus.

The hash check is the subject of most of these: it is the only thing
standing between a re-parse and a dataset whose labels silently point at
neighbouring text while still producing plausible numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from daedalus.config import constants
from daedalus.evaluation.corpus import (
    _read_text,
    _write_text,
    freeze_corpus,
    load_frozen,
    load_manifest,
    text_path_for,
)


@pytest.fixture
def eval_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    parsed = tmp_path / "parsed"
    monkeypatch.setattr(constants, "EVAL_PARSED_DIR", parsed)
    monkeypatch.setattr(constants, "EVAL_MANIFEST_PATH", parsed / "manifest.json")

    return parsed


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    source = tmp_path / "corpus" / "arXiv"
    source.mkdir(parents=True)
    (source / "paper.md").write_text("# Paper\n\nScaled dot-product attention.\n", encoding="utf-8")

    return tmp_path / "corpus"


def test_a_document_is_frozen(eval_dir: Path, corpus: Path) -> None:
    frozen = freeze_corpus(corpus)

    assert [document.doc_id for document in frozen] == ["paper"]
    assert text_path_for("paper").exists()


def test_the_source_type_comes_from_the_corpus_folder(eval_dir: Path, corpus: Path) -> None:
    """arxiv and course_notes are both PDFs; the folder is what separates them."""

    assert freeze_corpus(corpus)[0].source_type == "arxiv"


def test_the_manifest_records_what_was_frozen(eval_dir: Path, corpus: Path) -> None:
    freeze_corpus(corpus)

    manifest = load_manifest()

    assert manifest["version"] == 1
    assert manifest["documents"][0]["doc_id"] == "paper"  # type: ignore[index]


def test_frozen_text_reloads_unchanged(eval_dir: Path, corpus: Path) -> None:
    original = freeze_corpus(corpus)[0]

    assert load_frozen()[0].text == original.text


def test_carriage_returns_survive_the_round_trip(tmp_path: Path) -> None:
    """Regression: notebook outputs are full of \\r from progress bars.

    Python's text mode rewrites \\r and \\r\\n to \\n on read, which shortens
    the text and shifts every offset after the first one — silently
    invalidating every label anchored past it. The frozen files must be
    read and written byte-for-byte.
    """

    text = "epoch 1\rescaped 50%\rdone\r\nfinished"
    path = tmp_path / "frozen.txt"

    _write_text(path, text)

    assert _read_text(path) == text
    # What the obvious implementation would have done instead:
    assert path.read_text(encoding="utf-8") != text


def test_segments_survive_the_round_trip(eval_dir: Path, corpus: Path) -> None:
    """Without them, extraction method and page are lost and slicing dies."""

    original = freeze_corpus(corpus)[0]

    assert load_frozen()[0].segments == original.segments


def test_edited_text_is_caught(eval_dir: Path, corpus: Path) -> None:
    freeze_corpus(corpus)

    path = text_path_for("paper")
    path.write_text(path.read_text(encoding="utf-8") + " tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match the manifest hash"):
        load_frozen()


def test_an_unparseable_document_is_skipped_not_fatal(eval_dir: Path, tmp_path: Path) -> None:
    """CI has no vision model; a 90% frozen corpus is still usable."""

    source = tmp_path / "corpus" / "Images"
    source.mkdir(parents=True)
    (source / "diagram.png").write_bytes(b"not a real image")
    (source / "notes.md").write_text("real text", encoding="utf-8")

    frozen = freeze_corpus(tmp_path / "corpus")

    assert [document.doc_id for document in frozen] == ["notes"]


def test_unsupported_extensions_are_ignored(eval_dir: Path, corpus: Path) -> None:
    (corpus / "arXiv" / "notes.txt").write_text("plain text", encoding="utf-8")

    assert len(freeze_corpus(corpus)) == 1


def test_a_missing_manifest_says_how_to_make_one(eval_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"daedalus\.evaluation\.freeze"):
        load_frozen()


def test_a_frozen_document_rebuilds_the_parser_output(eval_dir: Path, corpus: Path) -> None:
    document = freeze_corpus(corpus)[0]
    parsed = document.as_parsed()

    assert parsed.doc_id == document.doc_id
    assert parsed.text == document.text
    assert parsed.source_type == document.source_type


def test_the_manifest_is_valid_json(eval_dir: Path, corpus: Path) -> None:
    freeze_corpus(corpus)

    json.loads(constants.EVAL_MANIFEST_PATH.read_text(encoding="utf-8"))

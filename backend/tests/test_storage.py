import hashlib
import io
import json

import pytest

from app.ingest.storage import FileTooLarge, UnsupportedFile, save_file

PDF = b"%PDF-1.4\n% tiny test file\n"
MB = 1024 * 1024


def test_files_are_stored_once_under_their_hash(tmp_path) -> None:
    first = save_file(tmp_path, io.BytesIO(PDF), "Notes.PDF", MB)
    second = save_file(tmp_path, io.BytesIO(PDF), "copy.pdf", MB)

    digest = hashlib.sha256(PDF).hexdigest()
    assert first == second
    assert first.path == f"uploads/{digest}.pdf"
    assert (first.source_type, first.size) == ("pdf", len(PDF))
    assert [p.name for p in (tmp_path / "uploads").iterdir()] == [f"{digest}.pdf"]


def test_a_notebook_must_be_json_with_cells(tmp_path) -> None:
    notebook = json.dumps({"cells": [], "metadata": {}, "nbformat": 4}).encode()
    assert save_file(tmp_path, io.BytesIO(notebook), "a.ipynb", MB).source_type == "notebook"

    for content in (b"not json", b'{"no": "cells"}'):
        with pytest.raises(UnsupportedFile):
            save_file(tmp_path, io.BytesIO(content), "b.ipynb", MB)


@pytest.mark.parametrize(
    ("content", "filename", "error"),
    [
        (b"hello", "notes.pdf", UnsupportedFile),
        (PDF, "notes.docx", UnsupportedFile),
        (PDF * 1000, "big.pdf", FileTooLarge),
    ],
)
def test_rejected_files_leave_nothing_behind(tmp_path, content, filename, error) -> None:
    with pytest.raises(error):
        save_file(tmp_path, io.BytesIO(content), filename, max_bytes=1000)
    uploads = tmp_path / "uploads"
    assert not uploads.exists() or not any(uploads.iterdir())

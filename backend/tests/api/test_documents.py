"""Upload and status endpoints, exercised end to end through the real pipeline."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from daedalus.config import constants, settings

Upload = Callable[..., httpx.Response]


# Accepting an Upload


def test_a_new_document_is_accepted(upload: Upload, document: Path) -> None:
    response = upload(document)

    assert response.status_code == 202


def test_the_response_identifies_the_document(upload: Upload, document: Path) -> None:
    body = upload(document).json()

    assert body["id"] == "attention"
    assert body["filename"] == "attention.md"
    assert body["source_type"] == "markdown"


def test_the_document_is_ingested(upload: Upload, document: Path, api: TestClient) -> None:
    """The whole slice: parsed, chunked, embedded, and indexed."""

    doc_id = upload(document).json()["id"]

    body = api.get(f"/documents/{doc_id}").json()

    assert body["status"] == constants.STATUS_COMPLETED
    assert body["n_chunks"] > 0
    assert body["error"] is None


def test_the_uploaded_file_is_kept(upload: Upload, document: Path) -> None:
    upload(document)

    assert (constants.UPLOAD_DIR / "attention.md").exists()


# Rejecting an Upload


def test_an_unsupported_type_is_refused(upload: Upload, tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("plain text is not a supported type", encoding="utf-8")

    assert upload(path).status_code == 415


def test_an_oversized_file_is_refused(
    upload: Upload, document: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "max_upload_mb", 0)

    assert upload(document).status_code == 413


def test_an_oversized_file_leaves_nothing_behind(
    upload: Upload, document: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The partial write must not survive the rejection."""

    monkeypatch.setattr(settings, "max_upload_mb", 0)

    upload(document)

    assert not (constants.UPLOAD_DIR / "attention.md").exists()


def test_a_traversing_filename_cannot_escape_the_upload_directory(
    upload: Upload, document: Path
) -> None:
    """The client controls the filename, so it is rebuilt rather than trusted."""

    response = upload(document, filename="../../../../tmp/escaped.md")

    assert response.status_code == 202

    stored = list(constants.UPLOAD_DIR.iterdir())

    assert [path.name for path in stored] == ["escaped.md"]


def test_a_document_id_collision_is_refused(upload: Upload, document: Path, tmp_path: Path) -> None:
    """Different content whose name slugs the same must not overwrite."""

    upload(document)

    other = tmp_path / "other.md"
    other.write_text("completely different material about kernels", encoding="utf-8")

    assert upload(other, filename="attention.md").status_code == 409


# Idempotency


def test_the_same_content_is_not_ingested_twice(upload: Upload, document: Path) -> None:
    first = upload(document)
    second = upload(document)

    assert first.status_code == 202
    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]


def test_the_same_content_under_a_new_name_is_recognised(upload: Upload, document: Path) -> None:
    """Identity is the bytes, not the filename."""

    original = upload(document).json()
    renamed = upload(document, filename="renamed.md")

    assert renamed.status_code == 200
    assert renamed.json()["id"] == original["id"]


# Status


def test_an_unknown_document_is_not_found(api: TestClient) -> None:
    assert api.get("/documents/nonexistent").status_code == 404


def test_a_failed_ingestion_is_reported(upload: Upload, tmp_path: Path, api: TestClient) -> None:
    """A file that cannot be parsed must end as failed, not stuck processing."""

    broken = tmp_path / "broken.ipynb"
    broken.write_text("this is not valid notebook json", encoding="utf-8")

    doc_id = upload(broken).json()["id"]

    body = api.get(f"/documents/{doc_id}").json()

    assert body["status"] == constants.STATUS_FAILED
    assert body["error"]

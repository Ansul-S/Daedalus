"""Tests for the document lifecycle."""

from __future__ import annotations

import sqlite3

import pytest

from daedalus.config import constants
from daedalus.core.exceptions import DocumentNotFoundError, DuplicateDocumentError
from daedalus.storage import documents


def _create(db: sqlite3.Connection, doc_id: str = "doc-1", content_hash: str = "hash-1") -> None:
    documents.create(
        db,
        doc_id=doc_id,
        filename=f"{doc_id}.pdf",
        source_type="arxiv",
        content_hash=content_hash,
    )


# Creation


def test_a_new_document_starts_pending(db: sqlite3.Connection) -> None:
    record = documents.create(
        db,
        doc_id="doc-1",
        filename="paper.pdf",
        source_type="arxiv",
        content_hash="hash-1",
    )

    assert record.status == constants.STATUS_PENDING
    assert record.filename == "paper.pdf"
    assert record.error is None
    assert record.n_chunks is None


def test_created_and_updated_timestamps_are_set(db: sqlite3.Connection) -> None:
    record = documents.create(
        db,
        doc_id="doc-1",
        filename="paper.pdf",
        source_type="arxiv",
        content_hash="hash-1",
    )

    assert record.created_at == record.updated_at
    assert record.created_at.endswith("+00:00"), "timestamps must be stored in UTC"


def test_the_same_content_cannot_be_ingested_twice(db: sqlite3.Connection) -> None:
    """Idempotency: two uploads of one file under different names are one document."""

    _create(db, "doc-1", content_hash="identical")

    with pytest.raises(DuplicateDocumentError):
        _create(db, "doc-2", content_hash="identical")


def test_the_same_id_cannot_be_reused(db: sqlite3.Connection) -> None:
    _create(db, "doc-1", content_hash="hash-1")

    with pytest.raises(DuplicateDocumentError):
        _create(db, "doc-1", content_hash="hash-2")


def test_a_rejected_duplicate_leaves_the_original_intact(db: sqlite3.Connection) -> None:
    _create(db, "doc-1", content_hash="identical")

    with pytest.raises(DuplicateDocumentError):
        _create(db, "doc-2", content_hash="identical")

    assert db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1


# Reads


def test_get_returns_none_for_an_unknown_document(db: sqlite3.Connection) -> None:
    assert documents.get(db, "never-uploaded") is None


def test_a_document_can_be_found_by_its_content(db: sqlite3.Connection) -> None:
    _create(db, "doc-1", content_hash="hash-1")

    found = documents.get_by_hash(db, "hash-1")

    assert found is not None
    assert found.id == "doc-1"


def test_get_by_hash_returns_none_for_unseen_content(db: sqlite3.Connection) -> None:
    assert documents.get_by_hash(db, "never-seen") is None


# Transitions


def test_status_moves_through_the_lifecycle(db: sqlite3.Connection) -> None:
    _create(db)

    documents.mark_processing(db, "doc-1")
    processing = documents.get(db, "doc-1")

    documents.mark_completed(db, "doc-1", n_pages=15, n_chunks=42)
    completed = documents.get(db, "doc-1")

    assert processing is not None
    assert completed is not None
    assert processing.status == constants.STATUS_PROCESSING
    assert completed.status == constants.STATUS_COMPLETED
    assert completed.n_pages == 15
    assert completed.n_chunks == 42


def test_failure_keeps_the_reason(db: sqlite3.Connection) -> None:
    _create(db)

    documents.mark_failed(db, "doc-1", "no text layer and OCR is unavailable")
    record = documents.get(db, "doc-1")

    assert record is not None
    assert record.status == constants.STATUS_FAILED
    assert record.error == "no text layer and OCR is unavailable"


def test_a_successful_retry_clears_the_previous_error(db: sqlite3.Connection) -> None:
    _create(db)
    documents.mark_failed(db, "doc-1", "transient failure")

    documents.mark_completed(db, "doc-1", n_chunks=3)
    record = documents.get(db, "doc-1")

    assert record is not None
    assert record.error is None


def test_completion_does_not_erase_counts_it_was_not_given(db: sqlite3.Connection) -> None:
    """COALESCE, not overwrite — a later transition must not blank n_pages."""

    _create(db)
    documents.mark_completed(db, "doc-1", n_pages=15, n_chunks=42)

    documents.mark_completed(db, "doc-1", n_chunks=44)
    record = documents.get(db, "doc-1")

    assert record is not None
    assert record.n_pages == 15
    assert record.n_chunks == 44


def test_updated_at_advances_on_a_transition(db: sqlite3.Connection) -> None:
    _create(db)
    before = documents.get(db, "doc-1")

    documents.mark_processing(db, "doc-1")
    after = documents.get(db, "doc-1")

    assert before is not None
    assert after is not None
    assert after.updated_at >= before.created_at
    assert after.created_at == before.created_at


@pytest.mark.parametrize(
    "transition",
    [
        lambda db: documents.mark_processing(db, "ghost"),
        lambda db: documents.mark_completed(db, "ghost", n_chunks=1),
        lambda db: documents.mark_failed(db, "ghost", "boom"),
    ],
)
def test_transitions_on_a_missing_document_are_loud(
    db: sqlite3.Connection,
    transition: object,
) -> None:
    """
    A typo in a document id must not be a silent no-op.

    UPDATE against zero rows is not an error in SQL, so without the rowcount
    check an ingestion task could report success having written nothing.
    """

    with pytest.raises(DocumentNotFoundError):
        transition(db)  # type: ignore[operator]


# Restart recovery


def test_stale_processing_documents_are_failed(db: sqlite3.Connection) -> None:
    """
    Background tasks do not survive a restart (ADR-008).

    A document left in ``processing`` has no task behind it, so a client
    polling its status would wait forever.
    """

    _create(db, "doc-1", content_hash="hash-1")
    _create(db, "doc-2", content_hash="hash-2")
    documents.mark_processing(db, "doc-1")

    reset = documents.reset_stale_processing(db)
    recovered = documents.get(db, "doc-1")
    untouched = documents.get(db, "doc-2")

    assert reset == 1
    assert recovered is not None
    assert recovered.status == constants.STATUS_FAILED
    assert recovered.error is not None
    assert untouched is not None
    assert untouched.status == constants.STATUS_PENDING


def test_resetting_with_nothing_stale_is_a_no_op(db: sqlite3.Connection) -> None:
    _create(db)
    documents.mark_completed(db, "doc-1", n_chunks=1)

    assert documents.reset_stale_processing(db) == 0

"""
Document upload and status.

Upload returns ``202 Accepted`` with a document id and hands the real work
to a background task, per ADR-008. The alternative — parsing, embedding,
and indexing inside the request — would hold the connection open for the
tens of seconds a large PDF takes and time out behind any proxy.

The client polls the status endpoint to find out how it went.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Response, UploadFile
from fastapi import status as http_status

from daedalus.api.dependencies import get_db, get_embedder
from daedalus.api.schemas import DocumentResponse
from daedalus.config import constants, settings
from daedalus.core.exceptions import DuplicateDocumentError
from daedalus.core.hashing import file_hash
from daedalus.ingestion.pipeline import ingest_document, source_type_for
from daedalus.ingestion.router import make_doc_id
from daedalus.interfaces.embedding import Embedder
from daedalus.storage import documents

__all__ = ["router"]


logger = logging.getLogger(__name__)


router = APIRouter(prefix="/documents", tags=["documents"])


_READ_SIZE = 1024 * 1024


def _stored_name(filename: str) -> str:
    """
    Build a safe filename from client-supplied text.

    The client controls this string, so it is rebuilt rather than trusted:
    ``../../etc/passwd`` would otherwise escape the upload directory
    entirely. Slugging the stem drops every separator, and the suffix is
    validated against the supported set before this is called.
    """

    original = Path(filename)

    return f"{make_doc_id(original)}{original.suffix.lower()}"


def _save(upload: UploadFile, target: Path, max_bytes: int) -> None:
    """Stream an upload to disk, refusing one that grows past the limit."""

    written = 0

    with target.open("wb") as handle:
        while block := upload.file.read(_READ_SIZE):
            written += len(block)

            # Checked while streaming rather than from Content-Length, which
            # a client can understate or omit.
            if written > max_bytes:
                handle.close()
                target.unlink(missing_ok=True)

                raise HTTPException(
                    status_code=http_status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"file exceeds the {settings.max_upload_mb} MB limit",
                )

            handle.write(block)


@router.post("", status_code=http_status.HTTP_202_ACCEPTED)
def upload_document(
    response: Response,
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File()],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> DocumentResponse:
    """
    Accept a document and schedule it for ingestion.

    Returns ``202`` for new content and ``200`` for content already
    indexed — re-uploading the same bytes is a no-op that reports the
    existing document rather than an error.
    """

    filename = file.filename or "document"
    suffix = Path(filename).suffix.lower()

    if suffix not in constants.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=http_status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"'{suffix}' is not supported "
                f"(expected one of {sorted(constants.SUPPORTED_EXTENSIONS)})"
            ),
        )

    target = constants.UPLOAD_DIR / _stored_name(filename)
    target.parent.mkdir(parents=True, exist_ok=True)

    _save(file, target, settings.max_upload_mb * 1024 * 1024)

    content_hash = file_hash(target)

    # Identity is the content, not the name. Checked before creating a row
    # so that a re-upload under a different filename is recognised.
    existing = documents.get_by_hash(db, content_hash)

    if existing is not None:
        logger.info("Upload of %s matches existing document %s", filename, existing.id)
        response.status_code = http_status.HTTP_200_OK

        return DocumentResponse.from_record(existing)

    doc_id = make_doc_id(Path(filename))

    try:
        record = documents.create(
            db,
            doc_id=doc_id,
            filename=filename,
            source_type=source_type_for(target),
            content_hash=content_hash,
        )
    except DuplicateDocumentError as error:
        # The content is new but the id is taken: two different files whose
        # names slug to the same thing. Loud rather than silently
        # overwriting somebody else's document.
        target.unlink(missing_ok=True)

        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail=f"a different document is already stored as {doc_id!r}",
        ) from error

    # Scheduled only after the row exists, so a status poll that arrives
    # before the task starts still finds the document.
    background_tasks.add_task(ingest_document, target, doc_id, embedder)

    return DocumentResponse.from_record(record)


@router.get("/{doc_id}")
def get_document(
    doc_id: str, db: Annotated[sqlite3.Connection, Depends(get_db)]
) -> DocumentResponse:
    """Report a document's ingestion status."""

    record = documents.get(db, doc_id)

    if record is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"no document with id {doc_id!r}",
        )

    return DocumentResponse.from_record(record)

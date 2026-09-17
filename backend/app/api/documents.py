"""Library endpoints: add study material and follow its ingestion.

Adding material only records a job; `make worker` (or `make ingest`) does the parsing.
Ingestion needs the local models, so these endpoints are disabled in production.
"""

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.models import Chunk, Document, Job
from app.db.session import get_session
from app.ingest import queue
from app.ingest.arxiv import parse_arxiv_id
from app.ingest.storage import FileTooLarge, UnsupportedFile, save_file

router = APIRouter(tags=["documents"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def local_only(settings: SettingsDep) -> None:
    if settings.environment != "local":
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "Ingestion runs locally; add material from your own machine"
        )


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    document_id: int
    status: str
    progress: str | None
    error: str | None
    attempts: int
    options: dict[str, Any]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    source_type: str
    title: str
    authors: str | None
    filename: str | None
    arxiv_id: str | None
    arxiv_version: str | None
    license: str | None
    url: str | None
    status: str
    error: str | None
    details: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    chunk_count: int = 0
    latest_job: JobOut | None = None


class IngestOut(BaseModel):
    message: str
    document: DocumentOut
    job: JobOut | None


class IngestOptions(BaseModel):
    ocr: bool = Field(False, description="Recognize text in scanned pages (PDF only)")
    formulas: bool = Field(True, description="Convert equations to LaTeX (PDF only; slower)")
    force: bool = Field(False, description="Ingest again even if the document is ready")


class ArxivIn(IngestOptions):
    arxiv_id: str = Field(examples=["1706.03762", "https://arxiv.org/abs/1706.03762v7"])


async def _documents_out(session: AsyncSession, documents: list[Document]) -> list[DocumentOut]:
    """Two queries for any number of documents: chunk counts and each latest job."""
    ids = [document.id for document in documents]
    counts = await session.execute(
        select(Chunk.document_id, func.count())
        .where(Chunk.document_id.in_(ids))
        .group_by(Chunk.document_id)
    )
    chunk_counts = dict(counts.tuples().all())
    latest_jobs = await session.scalars(
        select(Job)
        .where(Job.document_id.in_(ids))
        .distinct(Job.document_id)
        .order_by(Job.document_id, Job.id.desc())
    )
    jobs = {job.document_id: JobOut.model_validate(job) for job in latest_jobs}
    return [
        DocumentOut.model_validate(document).model_copy(
            update={
                "chunk_count": chunk_counts.get(document.id, 0),
                "latest_job": jobs.get(document.id),
            }
        )
        for document in documents
    ]


async def _ingest_out(
    session: AsyncSession, enqueued: queue.Enqueued, response: Response
) -> IngestOut:
    await session.commit()
    await session.refresh(enqueued.document)
    if enqueued.job is not None:
        await session.refresh(enqueued.job)
    if enqueued.job is None or enqueued.job.status != "queued":
        response.status_code = status.HTTP_200_OK
    [document] = await _documents_out(session, [enqueued.document])
    return IngestOut(
        message=enqueued.message,
        document=document,
        job=JobOut.model_validate(enqueued.job) if enqueued.job else None,
    )


@router.post(
    "/documents/upload",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(local_only)],
)
async def upload_document(
    file: UploadFile,
    options: Annotated[IngestOptions, Depends()],
    session: SessionDep,
    settings: SettingsDep,
    response: Response,
) -> IngestOut:
    """Upload a PDF or a Jupyter notebook. Returns 202 when a job was queued."""
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if file.size is not None and file.size > max_bytes:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE, f"file is larger than {settings.max_upload_mb} MB"
        )
    filename = file.filename or "upload"
    try:
        stored = await run_in_threadpool(
            save_file, settings.data_dir, file.file, filename, max_bytes
        )
    except FileTooLarge as exc:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, str(exc)) from exc
    except UnsupportedFile as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
    job_options = {"ocr": options.ocr, "formulas": options.formulas}
    enqueued = await queue.add_file(session, stored, filename, job_options, options.force)
    return await _ingest_out(session, enqueued, response)


@router.post(
    "/documents/arxiv",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(local_only)],
)
async def add_arxiv_paper(body: ArxivIn, session: SessionDep, response: Response) -> IngestOut:
    """Add an arXiv paper by ID or URL. Returns 202 when a job was queued."""
    try:
        arxiv_id, version = parse_arxiv_id(body.arxiv_id)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    job_options = {"ocr": body.ocr, "formulas": body.formulas}
    enqueued = await queue.add_arxiv(session, arxiv_id, version, job_options, body.force)
    return await _ingest_out(session, enqueued, response)


@router.get("/documents")
async def list_documents(session: SessionDep) -> list[DocumentOut]:
    documents = await session.scalars(select(Document).order_by(Document.created_at.desc()))
    return await _documents_out(session, list(documents))


@router.get("/documents/{document_id}")
async def get_document(document_id: int, session: SessionDep) -> DocumentOut:
    document = await session.get(Document, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    [out] = await _documents_out(session, [document])
    return out


@router.get("/jobs/{job_id}")
async def get_job(job_id: int, session: SessionDep) -> JobOut:
    job = await session.get(Job, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    return JobOut.model_validate(job)

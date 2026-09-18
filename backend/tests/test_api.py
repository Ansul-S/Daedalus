import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, update

from app.api.search import get_embedder
from app.db.models import Chunk, Document, Job
from app.main import app

NOTEBOOK = json.dumps({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}).encode()
PDF = b"%PDF-1.4\n% test file\n"


def upload(client: TestClient, filename: str, content: bytes, **options: str):
    return client.post("/documents/upload", params=options, files={"file": (filename, content)})


def test_an_upload_is_queued_once(client) -> None:
    first = upload(client, "Lesson 1.ipynb", NOTEBOOK)
    again = upload(client, "copy.ipynb", NOTEBOOK)

    assert first.status_code == 202
    body = first.json()
    assert body["message"] == "queued"
    assert (body["job"]["status"], body["job"]["options"]) == (
        "queued",
        {"ocr": False, "formulas": True},
    )
    document = body["document"]
    assert (document["title"], document["source_type"], document["status"]) == (
        "Lesson 1",
        "notebook",
        "pending",
    )
    assert (document["chunk_count"], document["latest_job"]["id"]) == (0, body["job"]["id"])
    # Still pending: the same job is returned.
    assert again.status_code == 202
    assert (again.json()["message"], again.json()["job"]["id"]) == (
        "already queued",
        body["job"]["id"],
    )


def test_an_ingested_file_is_only_queued_again_when_forced(client, sessions) -> None:
    first = upload(client, "lesson.ipynb", NOTEBOOK).json()

    async def finish() -> None:
        async with sessions() as session, session.begin():
            (await session.get_one(Job, first["job"]["id"])).status = "done"
            (await session.get_one(Document, first["document"]["id"])).status = "ready"

    asyncio.run(finish())
    again = upload(client, "lesson.ipynb", NOTEBOOK)
    forced = upload(client, "lesson.ipynb", NOTEBOOK, force="true")

    assert again.status_code == 200
    assert again.json()["job"] is None
    assert again.json()["message"].startswith("already ingested")
    assert forced.status_code == 202
    assert forced.json()["job"]["id"] != first["job"]["id"]


def test_upload_options_are_stored_with_the_job(client) -> None:
    response = upload(client, "scan.pdf", PDF, ocr="true", formulas="false")

    assert response.status_code == 202
    assert response.json()["job"]["options"] == {"ocr": True, "formulas": False}


@pytest.mark.parametrize(
    ("filename", "content", "status_code", "detail"),
    [
        ("notes.docx", PDF, 415, "only PDF (.pdf) and Jupyter notebook (.ipynb) files"),
        ("notes.pdf", b"<html></html>", 415, "not a PDF"),
        ("lesson.ipynb", b'{"cells": 1}', 415, "not a Jupyter notebook"),
        ("big.pdf", PDF + b"0" * 1024 * 1024, 413, "larger than 1 MB"),
    ],
)
def test_unsupported_and_large_files_are_rejected(
    client, settings, filename, content, status_code, detail
) -> None:
    response = upload(client, filename, content)

    assert response.status_code == status_code
    assert detail in response.json()["detail"]
    assert client.get("/documents").json() == []
    uploads = settings.data_dir / "uploads"
    assert not uploads.exists() or not any(uploads.iterdir())


def test_adding_material_is_disabled_in_production(client, settings) -> None:
    settings.environment = "production"

    assert upload(client, "lesson.ipynb", NOTEBOOK).status_code == 403
    assert client.post("/documents/arxiv", json={"arxiv_id": "1706.03762"}).status_code == 403
    assert client.get("/documents").status_code == 200


def test_arxiv_papers_are_added_by_id_or_url(client) -> None:
    bad = client.post("/documents/arxiv", json={"arxiv_id": "attention is all you need"})
    good = client.post(
        "/documents/arxiv",
        json={"arxiv_id": "https://arxiv.org/abs/1706.03762v7", "formulas": False},
    )

    assert bad.status_code == 422
    assert "not an arXiv ID" in bad.json()["detail"]
    assert good.status_code == 202
    body = good.json()
    assert (body["document"]["arxiv_id"], body["document"]["title"]) == (
        "1706.03762",
        "arXiv:1706.03762",
    )
    assert body["job"]["options"] == {"ocr": False, "formulas": False, "version": "v7"}


def test_documents_show_their_chunk_count_and_latest_job(client, sessions, corpus) -> None:
    async def add_jobs() -> tuple[int, int, int]:
        async with sessions() as session, session.begin():
            paper = await session.scalar(select(Document).where(Document.arxiv_id == "1706.03762"))
            finished = Job(document_id=paper.id, status="done", progress="3 chunks")
            session.add(finished)
            await session.flush()
            queued = Job(document_id=paper.id)
            session.add(queued)
            await session.flush()
            return paper.id, finished.id, queued.id

    paper_id, finished_id, queued_id = asyncio.run(add_jobs())

    listed = {document["id"]: document for document in client.get("/documents").json()}
    detail = client.get(f"/documents/{paper_id}").json()
    assert len(listed) == 3
    assert (listed[paper_id]["chunk_count"], listed[paper_id]["latest_job"]["id"]) == (3, queued_id)
    assert (detail["title"], detail["chunk_count"]) == ("Attention Is All You Need", 3)
    assert client.get(f"/jobs/{finished_id}").json()["progress"] == "3 chunks"
    assert client.get("/documents/999").status_code == 404
    assert client.get("/jobs/999").status_code == 404


def test_a_job_that_writes_questions_belongs_to_no_document(client, sessions) -> None:
    async def add() -> int:
        async with sessions() as session, session.begin():
            job = Job(kind="generate", options={"count": 5, "planned": 5})
            session.add(job)
            await session.flush()
            return job.id

    job_id = asyncio.run(add())

    body = client.get(f"/jobs/{job_id}").json()
    assert (body["kind"], body["document_id"]) == ("generate", None)
    assert body["options"]["planned"] == 5


def test_superseded_chunks_are_left_out_of_the_chunk_count(client, sessions, corpus) -> None:
    async def supersede() -> int:
        async with sessions() as session, session.begin():
            paper = await session.scalar(select(Document).where(Document.arxiv_id == "1706.03762"))
            await session.execute(
                update(Chunk).where(Chunk.id == corpus.scaling).values(superseded_at=func.now())
            )
            return paper.id

    paper_id = asyncio.run(supersede())

    assert client.get(f"/documents/{paper_id}").json()["chunk_count"] == 2


@pytest.mark.parametrize(
    ("query", "citation", "link"),
    [
        (
            "why divide by the square root of the key dimension",
            "Attention Is All You Need, § 3.2.1 Scaled Dot-Product Attention",
            "https://arxiv.org/html/1706.03762v7#S3.SS2.SSS1",
        ),
        ("vanishing gradients", "RNN Intuition, pp. 9–10", None),
        ("what does FAISS return", "Retrieval QnA, cells 12–14", None),
    ],
)
def test_search_results_carry_citations_and_links(client, corpus, query, citation, link) -> None:
    response = client.get("/search", params={"q": query, "limit": 3})

    assert response.status_code == 200
    body = response.json()
    assert (body["query"], body["mode"], body["warning"]) == (query, "hybrid", None)
    assert len(body["results"]) == 3
    top = body["results"][0]
    assert (top["citation"], top["link"]) == (citation, link)
    assert (top["vector_rank"], top["keyword_rank"]) == (1, 1)


def test_search_without_the_embedding_model(client, corpus) -> None:
    app.dependency_overrides[get_embedder] = lambda: None

    vector = client.get("/search", params={"q": "softmax", "mode": "vector"})
    hybrid = client.get("/search", params={"q": "softmax"}).json()

    assert vector.status_code == 503
    assert "local embedding model" in vector.json()["detail"]
    assert (hybrid["mode"], hybrid["results"][0]["chunk_id"]) == ("keyword", corpus.softmax)
    assert hybrid["warning"].endswith("showing keyword matches only")


@pytest.mark.parametrize(
    "params", [{"q": ""}, {"q": "softmax", "limit": 0}, {"q": "softmax", "mode": "fuzzy"}]
)
def test_invalid_search_parameters_are_rejected(client, params) -> None:
    assert client.get("/search", params=params).status_code == 422

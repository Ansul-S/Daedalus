import asyncio
import io
from pathlib import Path

import httpx
import pytest

nbformat = pytest.importorskip("nbformat")
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.db.models import Chunk, Document, Job  # noqa: E402
from app.ingest import arxiv, pipeline, queue  # noqa: E402
from app.ingest.arxiv import ArxivClient  # noqa: E402
from app.ingest.pipeline import Ingestor  # noqa: E402
from app.ingest.storage import StoredFile, save_file  # noqa: E402

OPTIONS = {"ocr": False, "formulas": True}
CELLS = [
    new_markdown_cell("# Retrieval QnA"),
    new_markdown_cell("## Retriever\nFAISS finds the chunks closest to the question."),
    new_code_cell("index = faiss.IndexFlatIP(384)"),
    new_markdown_cell("## Reader\nA BERT reader extracts the answer span."),
]
PAPER_URL = "https://arxiv.org/html/1706.03762v7"
RECORD = """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><GetRecord><record><metadata>
  <arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/">
    <id>1706.03762</id>
    <version version="v1"/><version version="v7"/>
    <title>Attention Is All You Need</title>
    <authors>Ashish Vaswani, Noam Shazeer</authors>
    <categories>cs.CL cs.LG</categories>
    <license>http://arxiv.org/licenses/nonexclusive-distrib/1.0/</license>
    <abstract>The dominant sequence transduction models...</abstract>
  </arXivRaw>
</metadata></record></GetRecord></OAI-PMH>"""


@pytest.fixture
def settings(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    # Chunk sizes are counted in words, so no tokenizer is downloaded.
    monkeypatch.setattr(pipeline, "token_counter", lambda model: lambda text: len(text.split()))
    monkeypatch.setattr(arxiv, "REQUEST_INTERVAL", 0.0)
    return Settings(_env_file=None, data_dir=tmp_path, chunk_min_tokens=5, chunk_max_tokens=50)


def save_notebook(settings: Settings) -> StoredFile:
    content = nbformat.writes(new_notebook(cells=CELLS)).encode()
    return save_file(settings.data_dir, io.BytesIO(content), "lesson.ipynb", max_bytes=10**6)


def no_network(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected request to {request.url}")


async def run_queue(settings, sessions, embedder, handler=no_network) -> list[str]:
    """Process all queued jobs and return the progress messages."""
    messages: list[str] = []
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with ArxivClient(settings.data_dir / "arxiv", http=http) as client:
        ingestor = Ingestor(
            settings, sessions, embedder, client, lambda _, message: messages.append(message)
        )
        await ingestor.process_queue(stop_when_empty=True)
    return messages


async def add_notebook(sessions, stored: StoredFile, force: bool = False) -> queue.Enqueued:
    async with sessions() as session:
        added = await queue.add_file(session, stored, "lesson.ipynb", OPTIONS, force)
        await session.commit()
    return added


async def load(sessions, added: queue.Enqueued) -> tuple[Document, Job, list[Chunk]]:
    async with sessions() as session:
        document = await session.get_one(Document, added.document.id)
        job = await session.get_one(Job, added.job.id)
        chunks = await session.scalars(select(Chunk).order_by(Chunk.position))
        return document, job, list(chunks)


def test_a_notebook_is_parsed_embedded_and_stored(settings, sessions, embedder) -> None:
    stored = save_notebook(settings)

    async def scenario():
        added = await add_notebook(sessions, stored)
        messages = await run_queue(settings, sessions, embedder)
        return messages, *await load(sessions, added)

    messages, document, job, chunks = asyncio.run(scenario())

    assert [(c.section, c.content_types, c.cell_start, c.cell_end) for c in chunks] == [
        ("Retriever", ["text", "code"], 2, 3),
        ("Reader", ["text"], 4, 4),
    ]
    assert chunks[0].text == (
        "FAISS finds the chunks closest to the question.\n\n"
        "```python\nindex = faiss.IndexFlatIP(384)\n```"
    )
    assert chunks[0].token_count == len(chunks[0].text.split())
    assert (chunks[0].page_start, chunks[0].anchor) == (None, None)
    # The title and section path are embedded along with the text.
    assert embedder.documents[0] == f"Retrieval QnA > Retriever\n\n{chunks[0].text}"
    assert chunks[0].embedding == pytest.approx(embedder.vector(embedder.documents[0]), abs=1e-3)
    assert (document.status, document.title, document.error) == ("ready", "Retrieval QnA", None)
    assert document.details["chunks"] == 2
    assert document.details["tokens"] == sum(chunk.token_count for chunk in chunks)
    assert {"cells", "language", "parse_seconds", "embed_seconds"} <= document.details.keys()
    assert (job.status, job.progress, job.error) == ("done", "2 chunks", None)
    assert job.finished_at is not None
    assert messages[:2] == ["parsing", "embedding 2/2 chunks"]
    assert messages[-1].startswith("done: 2 chunks")


def test_a_file_that_cannot_be_parsed_fails(settings, sessions, embedder) -> None:
    stored = save_notebook(settings)
    (settings.data_dir / stored.path).write_text("{ not a notebook")

    async def scenario():
        added = await add_notebook(sessions, stored)
        await run_queue(settings, sessions, embedder)
        return await load(sessions, added)

    document, job, chunks = asyncio.run(scenario())

    assert job.status == "failed"
    assert job.error.startswith("NotJSONError: Notebook does not appear to be JSON")
    assert (document.status, document.error) == ("failed", job.error)
    assert chunks == []
    assert embedder.documents == []


def test_ingesting_again_replaces_the_chunks_unless_it_fails(settings, sessions, embedder) -> None:
    stored = save_notebook(settings)

    async def chunk_ids() -> list[int]:
        async with sessions() as session:
            return list(await session.scalars(select(Chunk.id).order_by(Chunk.position)))

    async def scenario():
        await add_notebook(sessions, stored)
        await run_queue(settings, sessions, embedder)
        first = await chunk_ids()
        await add_notebook(sessions, stored, force=True)
        await run_queue(settings, sessions, embedder)
        second = await chunk_ids()
        embedder.fail = True
        failed = await add_notebook(sessions, stored, force=True)
        await run_queue(settings, sessions, embedder)
        return first, second, await chunk_ids(), *await load(sessions, failed)

    first, second, after_failure, document, job, _ = asyncio.run(scenario())

    assert len(first) == len(second) == 2
    assert set(first).isdisjoint(second)
    # A failed run leaves the searchable chunks and the ready status alone.
    assert after_failure == second
    assert (document.status, document.error) == ("ready", None)
    assert job.status == "failed"
    assert job.error.startswith("EmbeddingError: Ollama is not reachable")


def test_an_arxiv_paper_is_read_from_its_html_page(settings, sessions, embedder) -> None:
    pytest.importorskip("bs4")
    pytest.importorskip("docling")
    page = (Path(__file__).parent / "fixtures" / "latexml_paper.html").read_text(encoding="utf-8")
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(f"{request.url.host}{request.url.path}")
        if request.url.host == "oaipmh.arxiv.org":
            return httpx.Response(200, text=RECORD)
        if request.url == PAPER_URL:
            return httpx.Response(200, text=page)
        return httpx.Response(404)

    async def scenario():
        async with sessions() as session:
            added = await queue.add_arxiv(session, "1706.03762", None, OPTIONS)
            await session.commit()
        await run_queue(settings, sessions, embedder, handler)
        return await load(sessions, added)

    document, job, chunks = asyncio.run(scenario())

    assert requested == ["oaipmh.arxiv.org/oai", "arxiv.org/html/1706.03762v7"]
    assert job.status == "done", job.error
    assert (document.status, document.title) == ("ready", "Attention Is All You Need")
    assert document.authors == "Ashish Vaswani, Noam Shazeer"
    assert document.license == "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"
    assert (document.arxiv_version, document.url) == ("v7", PAPER_URL)
    assert document.details["source"] == PAPER_URL
    assert document.details["categories"] == ["cs.CL", "cs.LG"]
    assert [(c.section, c.anchor, c.content_types) for c in chunks] == [
        ("Abstract", "abstract1", ["text"]),
        ("1 Introduction", "S1", ["text", "formula"]),
        ("3 Model Architecture", "S3", ["text", "formula", "table"]),
    ]
    assert "## 3.2 Attention" in chunks[2].text
    assert all(chunk.page_start is None and chunk.cell_start is None for chunk in chunks)
    assert (settings.data_dir / "arxiv" / "1706.03762" / "v7.html").exists()

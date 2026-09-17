"""Shared fixtures.

Database tests use their own database, `daedalus_test`, on the Postgres server from
`make db-up`. It is created and migrated once per test run and dropped afterwards, so the
development database is never touched. Without a running server those tests are skipped.
"""

import asyncio
import math
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.core.config import get_settings
from app.db.models import EMBEDDING_DIMENSIONS, Chunk, Document
from app.llm.embeddings import EmbeddingError, Progress

BACKEND = Path(__file__).resolve().parents[1]
TEST_DATABASE = "daedalus_test"


class FakeEmbedder:
    """Stands in for Ollama. Every distinct word gets its own dimension, so texts that share
    words are close and the ranking of a test corpus is predictable."""

    def __init__(self) -> None:
        self.fail = False
        self.documents: list[str] = []
        self._dimensions: dict[str, int] = {}

    def vector(self, value: str) -> list[float]:
        counts = [0.0] * EMBEDDING_DIMENSIONS
        for word in re.findall(r"[a-z0-9]+", value.lower()):
            counts[self._dimensions.setdefault(word, len(self._dimensions))] += 1.0
        norm = math.sqrt(sum(count * count for count in counts))
        return [count / norm for count in counts]

    async def embed_query(self, query: str) -> list[float]:
        self._check()
        return self.vector(query)

    async def embed_documents(
        self, texts: Sequence[str], progress: Progress | None = None
    ) -> list[list[float]]:
        self._check()
        self.documents.extend(texts)
        if progress is not None:
            await progress(len(texts), len(texts))
        return [self.vector(value) for value in texts]

    def _check(self) -> None:
        if self.fail:
            raise EmbeddingError("Ollama is not reachable (ConnectError)")


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    server = make_url(get_settings().database_url)
    maintenance = server.set(drivername="postgresql", database="postgres")
    admin_url = maintenance.render_as_string(hide_password=False)
    try:
        admin = psycopg.connect(admin_url, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError:
        pytest.skip("Postgres is not running; start it with `make db-up`")
    with admin:
        admin.execute(f"DROP DATABASE IF EXISTS {TEST_DATABASE} WITH (FORCE)")
        admin.execute(f"CREATE DATABASE {TEST_DATABASE}")

    url = server.set(database=TEST_DATABASE).render_as_string(hide_password=False)
    config = Config(toml_file=BACKEND / "pyproject.toml", attributes={"database_url": url})
    command.upgrade(config, "head")
    yield url

    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(f"DROP DATABASE IF EXISTS {TEST_DATABASE} WITH (FORCE)")


@pytest.fixture
def engine(database_url: str) -> Iterator[AsyncEngine]:
    """An engine on the emptied test database. Tests drive it with `asyncio.run`; without a
    connection pool, every event loop opens its own connections."""
    engine = create_async_engine(database_url, poolclass=NullPool)
    asyncio.run(_empty_tables(engine))
    yield engine
    asyncio.run(engine.dispose())


async def _empty_tables(engine: AsyncEngine) -> None:
    async with engine.begin() as connection:
        await connection.execute(text("TRUNCATE documents, chunks, jobs RESTART IDENTITY"))


@pytest.fixture
def sessions(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


@dataclass
class Corpus:
    """Chunk ids of the search corpus, by topic."""

    scaling: int
    positions: int
    softmax: int
    vanishing: int
    retriever: int


@pytest.fixture
def corpus(sessions: async_sessionmaker[AsyncSession], embedder: FakeEmbedder) -> Corpus:
    """One arXiv paper, one PDF and one notebook, embedded with the fake embedder."""
    paper = Document(
        source_type="arxiv",
        title="Attention Is All You Need",
        arxiv_id="1706.03762",
        url="https://arxiv.org/html/1706.03762v7",
        status="ready",
    )
    notes = Document(
        source_type="pdf",
        title="RNN Intuition",
        filename="rnn.pdf",
        sha256="a" * 64,
        status="ready",
    )
    notebook = Document(
        source_type="notebook",
        title="Retrieval QnA",
        filename="qna.ipynb",
        sha256="b" * 64,
        status="ready",
    )
    rows = [
        (
            paper,
            "3 Model Architecture > 3.2.1 Scaled Dot-Product Attention",
            "We divide the dot products by the square root of the key dimension, "
            "which keeps the softmax gradients large.",
            {"anchor": "S3.SS2.SSS1"},
        ),
        (
            paper,
            "3 Model Architecture > 3.5 Positional Encoding",
            "Sinusoidal encodings let the model attend by relative position.",
            {"anchor": "S3.SS5"},
        ),
        (
            paper,
            "Softmax Temperature",
            "Dividing logits by a temperature flattens the distribution.",
            {},
        ),
        (
            notes,
            "9 Limitations of Simple RNN > 9.2 Vanishing Gradient Problem",
            "Gradients shrink as they flow back through many time steps.",
            {"page_start": 9, "page_end": 10},
        ),
        (
            notebook,
            "Retriever",
            "FAISS returns the chunks closest to the question embedding.",
            {"cell_start": 12, "cell_end": 14},
        ),
    ]

    async def add() -> list[int]:
        async with sessions() as session, session.begin():
            session.add_all([paper, notes, notebook])
            await session.flush()
            chunks = [
                Chunk(
                    document_id=document.id,
                    position=position,
                    section=section,
                    content_types=["text"],
                    text=body,
                    token_count=len(body.split()),
                    embedding=embedder.vector(f"{section}\n\n{body}"),
                    **locators,
                )
                for position, (document, section, body, locators) in enumerate(rows)
            ]
            session.add_all(chunks)
            await session.flush()
            return [chunk.id for chunk in chunks]

    return Corpus(*asyncio.run(add()))

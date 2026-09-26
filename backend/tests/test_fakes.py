"""The stand-ins of FAKE_MODELS: each answers in its model's schema, and the pipeline keeps
what they write, all the way to a graded answer, as the end-to-end test relies on."""

import asyncio
import json
import math
import os
import subprocess
import sys

import httpx
import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.db.models import Chunk, Document
from app.grading.grader import grade_answer
from app.ingest.arxiv import ArxivClient
from app.ingest.pipeline import Ingestor
from app.llm import fakes
from app.llm.embeddings import query_input
from app.llm.models import (
    embedding_model,
    generation_model,
    grading_model,
    helper_model,
    paced_generation_model,
)
from app.main import app
from app.questions.generation import Source, generate_question
from app.questions.tagging import read_chunk, tagger
from app.questions.validation import store_question, validate
from tests.conftest import BACKEND

TITLE = "Training notes"
RESIDUAL = (
    "A residual connection adds the input of a block to its output, so the block only has "
    "to learn a correction to the identity. Because the identity path carries the signal "
    "unchanged, gradients reach the early layers without shrinking through every "
    "nonlinearity. This is why networks with a hundred layers still train when their blocks "
    "are residual."
)
NORMALIZATION = (
    "Layer normalization rescales the activations of each example to zero mean and unit "
    "variance across its features. Unlike batch normalization it needs no statistics from "
    "other examples, so it behaves the same at training time and at inference. Transformers "
    "apply it around every attention and feed-forward block."
)
WARMUP = (
    "Learning rate warmup starts training with a small learning rate and raises it over the "
    "first few thousand steps. Early on the optimizer's estimates of the gradient's scale are "
    "poor, and a full-size step can throw the weights far from where training began. Once "
    "those estimates settle, the schedule hands over to its usual decay."
)
SECTIONS = {
    "Residual connections": RESIDUAL,
    "Layer normalization": NORMALIZATION,
    "Learning rate warmup": WARMUP,
}


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Fake models, and passages cut small enough for a notebook of three short sections."""
    return Settings(_env_file=None, data_dir=tmp_path, fake_models=True, chunk_min_tokens=20)


def test_fake_models_are_refused_in_production() -> None:
    with pytest.raises(ValidationError, match="FAKE_MODELS") as refused:
        Settings(
            _env_file=None,
            environment="production",
            fake_models=True,
            groq_api_key="gsk_not-a-real-key",
        )
    # The refusal names the setting alone, never the others it was given
    assert "not-a-real-key" not in str(refused.value)


def test_every_model_is_a_fake_when_fake_models_is_on(settings: Settings) -> None:
    # A day's budget spent in full makes no difference to a stand-in
    spent = {fakes.WRITER: (10_000, 10_000_000)}

    assert generation_model(settings).model_name == "fake-writer"
    assert paced_generation_model(settings, spent).model_name == "fake-writer"
    assert grading_model(settings, spent).model_name == "fake-grader"
    assert helper_model(settings).model_name == "fake-helper"

    async def embed() -> tuple[list[list[float]], list[float]]:
        async with embedding_model(settings) as embedder:
            documents = await embedder.embed_documents(["residual connections"])
            return documents, await embedder.embed_query("why do residual connections help?")

    documents, query = asyncio.run(embed())
    assert documents == [fakes.vector("residual connections")]
    assert query == fakes.vector(query_input("why do residual connections help?"))


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_fake_vectors_are_the_same_in_every_process() -> None:
    """The worker embeds the passages and the API the queries: they have to agree."""
    text = "Residual connections carry gradients to the early layers."
    script = f"import json; from app.llm.fakes import vector; print(json.dumps(vector({text!r})))"
    elsewhere = subprocess.run(
        [sys.executable, "-c", script],
        cwd=BACKEND,
        env={**os.environ, "PYTHONHASHSEED": "4242"},
        capture_output=True,
        text=True,
        check=True,
    )
    vector = fakes.vector(text)

    assert json.loads(elsewhere.stdout) == vector
    assert math.isclose(dot(vector, vector), 1.0)
    shared = dot(vector, fakes.vector("gradients reach the early layers"))
    unrelated = dot(vector, fakes.vector("warmup raises the learning rate"))
    assert shared > unrelated
    assert fakes.vector("") == fakes.vector("---")


def test_the_fake_tagger_names_the_passage_s_concepts() -> None:
    agent = tagger(fakes.helper())
    prose = Chunk(
        id=1, position=1, section="Residual connections", content_types=["text"], text=RESIDUAL
    )
    code = Chunk(
        id=2,
        position=2,
        section="Residual connections",
        content_types=["code"],
        text="```python\nx = x + block(x)\n```",
    )

    untitled = Chunk(id=3, position=4, section=None, content_types=["text"], text=WARMUP)
    spanning = Chunk(
        id=4,
        position=5,
        section="Normalization > Layer norm · RMS norm",
        content_types=["text"],
        text=NORMALIZATION,
    )

    read = asyncio.run(read_chunk(agent, TITLE, prose))
    code_read = asyncio.run(read_chunk(agent, TITLE, code))

    assert (read.tags, read.explains, read.worth_asking) == (
        ["residual connections"],
        "residual connections",
        True,
    )
    # Nothing to quote in code: the stand-in says so, and so do the rules
    assert not code_read.model_worth_asking
    assert code_read.skip_reason == "code only"
    assert asyncio.run(read_chunk(agent, TITLE, spanning)).tags == ["layer norm", "rms norm"]
    # Without a section, the words it uses most
    assert asyncio.run(read_chunk(agent, TITLE, untitled)).tags == [
        "learning",
        "rate",
        "training",
    ]


@pytest.fixture
def passages(sessions) -> list[int]:
    """Two passages of a notebook, stored with their stand-in vectors."""

    async def add() -> list[int]:
        async with sessions() as session, session.begin():
            notebook = Document(
                source_type="notebook",
                title=TITLE,
                filename="notes.ipynb",
                sha256="c" * 64,
                status="ready",
            )
            session.add(notebook)
            await session.flush()
            chunks = [
                Chunk(
                    document_id=notebook.id,
                    position=position,
                    section=section,
                    content_types=["text"],
                    text=text,
                    token_count=fakes.count_tokens(text),
                    cell_start=position + 2,
                    cell_end=position + 2,
                    embedding=fakes.vector(text),
                )
                for position, (section, text) in enumerate(
                    [("Residual connections", RESIDUAL), ("Layer normalization", NORMALIZATION)]
                )
            ]
            session.add_all(chunks)
            await session.flush()
            return [chunk.id for chunk in chunks]

    return asyncio.run(add())


def test_a_fake_question_is_grounded_and_passes_every_check(
    sessions, passages: list[int], settings: Settings
) -> None:
    residual = [Source(passages[0], f"{TITLE} > Residual connections", RESIDUAL)]
    normalization = [Source(passages[1], f"{TITLE} > Layer normalization", NORMALIZATION)]

    async def scenario():
        async with embedding_model(settings) as embedder, sessions() as session:

            async def check(sources: list[Source], style: str):
                written = await generate_question(fakes.writer(), sources, style)
                checked = await validate(
                    session,
                    written,
                    sources,
                    model=fakes.helper(),
                    embedder=embedder,
                    similarity=settings.duplicate_similarity,
                )
                return written, checked

            first, first_checked = await check(residual, "why_how")
            await store_question(session, first, first_checked, residual)
            await session.commit()
            # The next question, from another passage, is judged against the first one
            return first, first_checked, *await check(normalization, "intuition")

    first, first_checked, second, second_checked = asyncio.run(scenario())

    assert first.question.question.startswith("Why is it that a residual connection adds")
    assert first.grounded and first.attempts == 1
    assert first.model == "fake-writer"
    assert 2 <= len(first.question.key_points) <= 4
    assert first_checked.passed, first_checked.failed
    report = first_checked.report
    assert (report["kind"], report["answerable"], report["checker_model"]) == (
        "explain",
        True,
        "fake-helper",
    )
    assert second_checked.passed, second_checked.failed
    assert second_checked.report["nearest_question"] is not None
    assert second_checked.report["nearest_similarity"] < settings.duplicate_similarity


def test_the_fake_grader_labels_by_word_overlap_and_cites_the_passage() -> None:
    sources = [Source(7, f"{TITLE} > Residual connections", RESIDUAL)]
    first, second, third = fakes.quotable(RESIDUAL)
    key_points = [
        {"text": first, "weight": 2, "evidence_quote": first, "chunk_id": 7},
        {"text": second, "weight": 1, "evidence_quote": second, "chunk_id": 7},
        {"text": third, "weight": 1, "evidence_quote": third, "chunk_id": 7},
    ]
    answer = (
        f"{first} The identity path carries gradients to the early layers. "
        "Such models are also cheaper to run on phones."
    )

    graded = asyncio.run(
        grade_answer(
            fakes.grader(), "Why do residual connections help?", key_points, sources, answer
        )
    )

    assert graded.model == "fake-grader"
    # All of the first point, about half the words of the second, next to none of the third
    assert [point["status"] for point in graded.key_points] == ["covered", "partial", "missing"]
    assert graded.key_points[0]["quote_found"] is True
    assert [(claim["verdict"], claim["chunk_id"]) for claim in graded.claims] == [
        ("supported", 7),
        ("supported", 7),
        ("unverified", None),
    ]
    assert graded.result.score == 0.625
    assert graded.follow_up == f"The passage also says: “{second}” Why does that hold?"


def notebook_file() -> bytes:
    nbformat = pytest.importorskip("nbformat")
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

    cells = [new_markdown_cell(f"# {TITLE}")]
    for section, text in SECTIONS.items():
        cells.append(new_markdown_cell(f"## {section}\n{text}"))
        if section == "Residual connections":
            cells.append(new_code_cell("x = x + block(x)"))
    return nbformat.writes(new_notebook(cells=cells)).encode()


def no_network(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected request to {request.url}")


def test_a_notebook_becomes_a_cited_grade_on_fakes(
    client, sessions, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The backend half of `make e2e`: upload, topic map, a batch of three, an answer graded
    with a cited claim, and its room practised."""
    pytest.importorskip("sklearn")
    from scripts import worker

    notebook = notebook_file()
    # The worker's own code takes the jobs, on the test database
    monkeypatch.setattr(worker, "SessionFactory", sessions)
    # The grader is built as the API builds it, from the settings, and forgotten afterwards
    monkeypatch.setattr(app.state, "grader", None, raising=False)

    def work() -> None:
        async def take_every_job() -> None:
            http = httpx.AsyncClient(transport=httpx.MockTransport(no_network))
            async with (
                embedding_model(settings) as embedder,
                ArxivClient(settings.data_dir / "arxiv", http=http) as arxiv,
            ):
                ingestor = Ingestor(settings, sessions, embedder, arxiv, lambda *_: None)
                while await worker.take_a_job(settings, ingestor, embedder):
                    pass

        asyncio.run(take_every_job())

    upload = client.post(
        "/documents/upload", files={"file": ("notes.ipynb", notebook, "application/json")}
    )
    assert upload.status_code == 202, upload.text
    work()
    [document] = client.get("/documents").json()
    assert (document["status"], document["title"], document["chunk_count"]) == ("ready", TITLE, 3)

    assert client.post("/topics/build").status_code == 202
    work()
    assert client.get(f"/documents/{document['id']}").json()["tagged_count"] == 3

    batch = client.post("/questions/generate", json={"count": 3})
    assert batch.status_code == 202 and batch.json()["planned"] == 3
    work()
    written = client.get("/questions").json()
    assert written["total"] == 3
    assert {question["status"] for question in written["results"]} == {"accepted"}

    pick = client.get("/practice/next").json()["question"]
    answer = " ".join(" ".join(fakes.quotable(text)[:2]) for text in SECTIONS.values())
    attempt = client.post(f"/questions/{pick['id']}/attempts", json={"answer": answer})
    assert attempt.status_code == 201, attempt.text
    graded = attempt.json()
    [grade] = graded["grades"]
    assert (grade["status"], grade["grader_model"]) == ("graded", "fake-grader")
    cited = [claim for claim in grade["claims"] if claim["verdict"] == "supported"]
    assert cited and all(claim["citation"].startswith(f"{TITLE}, cell") for claim in cited)
    assert graded["review"]["rating"] in ("good", "easy")
    assert graded["earned"]["xp"] > 0

    rooms = client.get("/practice/map").json()["rooms"]
    assert [room["practised"] for room in rooms if room["name"] == pick["topic"]] == [1]

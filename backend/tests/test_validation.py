"""Judging a generated question, and writing down the verdict with the report behind it."""

import asyncio
import json

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile
from sqlalchemy import select

from app.db.models import Chunk, ChunkTopic, Document, Question, QuestionSource, Topic
from app.questions.generation import Generated, GeneratedQuestion, Source
from app.questions.grounding import QuoteCheck
from app.questions.topics import topic_for
from app.questions.validation import Validation, review, store_question, validate

QUESTION = "Why are the dot products scaled before the softmax?"
CHUNK = (
    "We suspect that for large values of $d_k$, the dot products grow large in magnitude, "
    "pushing the softmax function into regions where it has extremely small gradients."
)
GOOD_QUOTE = "the dot products grow large in magnitude, pushing the softmax function"
CITATION = "Attention Is All You Need > 3.2.1 Scaled Dot-Product Attention"


@pytest.fixture
def chunk_id(sessions, embedder) -> int:
    """One stored chunk, so that a question can point at something real."""

    async def add() -> int:
        async with sessions() as session, session.begin():
            paper = Document(
                source_type="arxiv",
                title="Attention Is All You Need",
                arxiv_id="1706.03762",
                status="ready",
            )
            paper.chunks = [
                Chunk(
                    position=0,
                    section="3.2.1 Scaled Dot-Product Attention",
                    content_types=["text"],
                    text=CHUNK,
                    token_count=len(CHUNK.split()),
                    embedding=embedder.vector(CHUNK),
                )
            ]
            session.add(paper)
            await session.flush()
            return paper.chunks[0].id

    return asyncio.run(add())


def sources_for(chunk_id: int) -> list[Source]:
    return [Source(chunk_id=chunk_id, citation=CITATION, text=CHUNK)]


def a_generated(chunk_id: int, question: str = QUESTION, problem: str | None = None) -> Generated:
    written = GeneratedQuestion(
        question=question,
        style="why_how",
        difficulty=3,
        reference_answer="Their magnitude grows with the key dimension, which flattens the "
        "softmax gradient.",
        key_points=[
            {
                "text": "large dot products saturate the softmax",
                "weight": 3,
                "evidence_quote": GOOD_QUOTE,
                "chunk_id": chunk_id,
            },
            {
                "text": "scaling undoes the growth",
                "weight": 2,
                "evidence_quote": GOOD_QUOTE,
                "chunk_id": chunk_id,
            },
        ],
        misconceptions=["It keeps the weights positive."],
        source_chunk_ids=[chunk_id],
    )
    return Generated(
        question=written,
        quotes=[
            QuoteCheck(quote=GOOD_QUOTE, chunk_id=chunk_id, score=100.0, problem=None),
            QuoteCheck(
                quote=GOOD_QUOTE,
                chunk_id=chunk_id,
                score=0.0 if problem else 100.0,
                problem=problem,
            ),
        ],
        model="groq:openai/gpt-oss-120b",
        usage={"requests": 1, "input_tokens": 1552, "output_tokens": 555},
    )


def a_checker(**overrides) -> FunctionModel:
    verdict = {
        "kind": "explain",
        "answer": "Because their magnitude grows with the key dimension.",
        "answerable": True,
        "missing": "nothing",
    } | overrides

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(json.dumps(verdict))])

    return FunctionModel(respond, profile=ModelProfile(supports_json_schema_output=True))


SIMILARITY = 0.7


def run_validate(sessions, embedder, chunk_id, generated=None, checker=None, **kwargs):
    async def scenario():
        async with sessions() as session:
            return await validate(
                session,
                generated or a_generated(chunk_id),
                sources_for(chunk_id),
                model=checker or a_checker(),
                embedder=embedder,
                similarity=kwargs.pop("similarity", SIMILARITY),
                **kwargs,
            )

    return asyncio.run(scenario())


def test_the_checker_is_shown_the_question_and_its_passages() -> None:
    prompt = review(QUESTION, sources_for(248))

    assert QUESTION in prompt
    assert f"[chunk 248] {CITATION}" in prompt
    assert f"<<<\n{CHUNK}\n>>>" in prompt


def test_a_question_that_holds_up_passes_every_check(sessions, embedder, chunk_id) -> None:
    validation = run_validate(sessions, embedder, chunk_id)

    assert (validation.passed, validation.failed) == (True, [])
    assert validation.report["kind"] == "explain"
    assert validation.report["answerable"] is True
    assert validation.report["missing"] == "nothing"
    assert validation.report["nearest_question"] is None
    assert [quote["score"] for quote in validation.report["quotes"]] == [100.0, 100.0]
    assert validation.embedding == pytest.approx(embedder.vector(QUESTION), abs=1e-6)
    # Recorded rather than judged: how far the reference answer drifts from the sources.
    assert 0.0 <= validation.report["answer_agreement"] <= 1.0


def test_a_question_the_passages_cannot_answer_is_turned_down(sessions, embedder, chunk_id):
    checker = a_checker(answerable=False, missing="it never says which optimizer was used")

    validation = run_validate(sessions, embedder, chunk_id, checker=checker)

    assert (validation.passed, validation.failed) == (False, ["answerable"])
    assert validation.report["missing"] == "it never says which optimizer was used"


def test_a_question_read_as_recall_is_turned_down_as_trivia(sessions, embedder, chunk_id):
    validation = run_validate(sessions, embedder, chunk_id, checker=a_checker(kind="recall"))

    assert (validation.passed, validation.failed) == (False, ["trivia"])


def test_a_quote_that_is_not_in_the_sources_fails_the_question(sessions, embedder, chunk_id):
    generated = a_generated(chunk_id, problem="the quote is not in the chunk (match 62%)")

    validation = run_validate(sessions, embedder, chunk_id, generated=generated)

    assert (validation.passed, validation.failed) == (False, ["quotes"])
    assert validation.report["quotes"][1]["problem"].startswith("the quote is not in the chunk")


def test_a_question_already_asked_in_other_words_is_turned_down(sessions, embedder, chunk_id):
    asked = Validation(passed=True, report={"failed": []}, embedding=embedder.vector(QUESTION))

    async def scenario():
        async with sessions() as session, session.begin():
            await store_question(session, a_generated(chunk_id), asked, sources_for(chunk_id))
        async with sessions() as session:
            return await validate(
                session,
                a_generated(
                    chunk_id, "Why are the dot products scaled before the softmax function?"
                ),
                sources_for(chunk_id),
                model=a_checker(),
                embedder=embedder,
                similarity=SIMILARITY,
            )

    validation = asyncio.run(scenario())

    assert (validation.passed, validation.failed) == (False, ["duplicate"])
    assert validation.duplicate_of == validation.report["nearest_question"]
    assert validation.report["nearest_similarity"] > 0.9


def test_a_different_question_about_the_same_chunk_is_kept(sessions, embedder, chunk_id) -> None:
    """The threshold has to separate a rewording from a second question about one passage."""
    asked = Validation(passed=True, report={"failed": []}, embedding=embedder.vector(QUESTION))

    async def scenario():
        async with sessions() as session, session.begin():
            await store_question(session, a_generated(chunk_id), asked, sources_for(chunk_id))
        async with sessions() as session:
            return await validate(
                session,
                a_generated(chunk_id, "How does FAISS index embeddings for retrieval?"),
                sources_for(chunk_id),
                model=a_checker(),
                embedder=embedder,
                similarity=SIMILARITY,
            )

    validation = asyncio.run(scenario())

    assert (validation.passed, validation.duplicate_of) == (True, None)
    assert validation.report["nearest_similarity"] < SIMILARITY


def test_without_the_embedding_model_the_duplicate_check_is_recorded_as_skipped(
    sessions, embedder, chunk_id
) -> None:
    embedder.fail = True

    validation = run_validate(sessions, embedder, chunk_id)

    assert (validation.passed, validation.embedding) == (True, None)
    assert validation.report["duplicate"] == "not checked"
    assert "answer_agreement" not in validation.report


def test_a_stored_question_keeps_its_sources_report_and_usage(sessions, embedder, chunk_id):
    async def scenario():
        generated = a_generated(chunk_id, problem="the quote is not in the chunk")
        sources = sources_for(chunk_id)
        async with sessions() as session:
            validation = await validate(
                session,
                generated,
                sources,
                model=a_checker(),
                embedder=embedder,
                similarity=SIMILARITY,
            )
        async with sessions() as session, session.begin():
            await store_question(session, generated, validation, sources)
        async with sessions() as session:
            question = await session.scalar(select(Question))
            stored = await session.scalars(select(QuestionSource.chunk_id))
            return question, list(stored)

    question, stored = asyncio.run(scenario())

    assert question.status == "rejected"
    assert question.validation["failed"] == ["quotes"]
    assert question.usage == {
        "requests": 1,
        "input_tokens": 1552,
        "output_tokens": 555,
        "attempts": 1,
    }
    assert (question.style, question.difficulty) == ("why_how", 3)
    assert [point["weight"] for point in question.key_points] == [3, 2]
    assert question.generator_model == "groq:openai/gpt-oss-120b"
    assert stored == [chunk_id]


def test_a_question_is_filed_under_the_topic_its_chunks_share(sessions, embedder, corpus) -> None:
    async def scenario():
        async with sessions() as session, session.begin():
            topics = [
                Topic(name=name, tags=[name], embedding=embedder.vector(name))
                for name in ("self attention", "softmax", "masked attention")
            ]
            session.add_all(topics)
            await session.flush()
            attention, softmax, masked = (topic.id for topic in topics)
            session.add_all(
                [
                    # Both chunks are about attention; each of the others covers only one.
                    ChunkTopic(chunk_id=corpus.scaling, topic_id=attention),
                    ChunkTopic(chunk_id=corpus.scaling, topic_id=softmax),
                    ChunkTopic(chunk_id=corpus.positions, topic_id=attention),
                    ChunkTopic(chunk_id=corpus.positions, topic_id=masked),
                    ChunkTopic(chunk_id=corpus.softmax, topic_id=attention),
                ]
            )
        async with sessions() as session:
            both = await topic_for(session, [corpus.scaling, corpus.positions])
            one = await topic_for(session, [corpus.softmax])
            none = await topic_for(session, [corpus.vanishing])
            return both, one, none, attention

    both, one, none, attention = asyncio.run(scenario())

    assert both == attention
    assert one == attention
    assert none is None

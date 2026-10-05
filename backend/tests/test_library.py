"""Putting the questions a review keeps into the library: only those that passed every check,
stored with their passages and a report the question page reads, a fixable one corrected as
the question bank corrects one."""

import asyncio
from datetime import date
from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select

from app.db.models import (
    Attempt,
    Card,
    Chunk,
    ChunkTopic,
    Document,
    Question,
    QuestionSource,
    Rating,
    Topic,
)
from app.questions.generation import Source
from app.questions.grounding import check_quote
from app.questions.inventory import Call
from app.questions.library import (
    Correction,
    Kept,
    Selection,
    correct,
    earlier,
    in_library,
    library_question,
    practised,
    store,
    unfit,
)
from app.questions.validation import AnswerCheck
from app.questions.writing import PROMPT_VERSION, Entry, IdeaQuestion
from scripts import store_questions

SATURATE = (
    "We suspect that for large values of $d_k$, the dot products grow large in magnitude, "
    "pushing the softmax function into regions where it has extremely small gradients."
)
SCALE = "To counteract this effect, we scale the dot products by $\\frac{1}{\\sqrt{d_k}}$."
HEADS = (
    "Multi-head attention allows the model to jointly attend to information from different "
    "representation subspaces at different positions."
)
TEXTS = [f"{SATURATE} {SCALE}", f"{HEADS} With a single attention head, averaging inhibits this."]
QUESTION = "Why are the dot products scaled before the softmax?"
LAYER_NORM = "Layer normalization rescales the activations of each example across its features."
READING = AnswerCheck(
    kind="explain",
    answer="Large dot products push the softmax where its gradients vanish.",
    answerable=True,
    missing="nothing",
)


def answer(**fields: Any) -> IdeaQuestion:
    values: dict[str, Any] = {
        "evidence": [{"sentence": SATURATE, "chunk_id": 1}, {"sentence": SCALE, "chunk_id": 1}],
        "question": QUESTION,
        "key_points": [
            {"text": "large dot products saturate the softmax", "weight": 3, "evidence": 1},
            {"text": "scaling undoes it", "weight": 2, "evidence": 2},
        ],
        "reference_answer": "Large dot products saturate the softmax; scaling undoes it.",
        "misconceptions": ["One", "Two", "Three", "Four"],
        "difficulty": 3,
    } | fields
    return IdeaQuestion.model_validate(values)


def entry(key: str = "1.1", chunk_ids: list[int] | None = None, **fields: Any) -> Entry:
    values: dict[str, Any] = {
        "key": key,
        "name": "scaling the dot products",
        "document_id": 1,
        "style": "why_how",
        "chunk_ids": chunk_ids if chunk_ids is not None else [1, 2],
        "answers": [answer()],
        "calls": [
            Call(
                model="openai/gpt-oss-120b",
                version=PROMPT_VERSION,
                usage={"requests": 1, "input_tokens": 900, "output_tokens": 300},
                at="2026-10-05T10:00:00+00:00",
            ),
            Call(
                model="openai/gpt-oss-120b",
                version=PROMPT_VERSION,
                usage={"requests": 1, "input_tokens": 1200, "output_tokens": 250},
                at="2026-10-05T10:01:00+00:00",
            ),
        ],
        "check": READING,
        "checker_model": "qwen3.5:4b",
    } | fields
    return Entry(**values)


def sources(chunk_ids: tuple[int, int] = (1, 2)) -> list[Source]:
    return [
        Source(chunk_id=chunk_id, citation="Attention Is All You Need", text=text)
        for chunk_id, text in zip(chunk_ids, TEXTS, strict=True)
    ]


def kept(key: str = "1.1", **fields: Any) -> Kept:
    values: dict[str, Any] = {
        "file": "library.questions.json",
        "key": key,
        "review": {"verdict": "good", "reason": "Asks why, as its passage explains.", "at": "t"},
    } | fields
    return Kept.model_validate(values)


def key_points(second_quote: str = SCALE) -> list[dict[str, Any]]:
    return [
        {
            "text": "Large dot products push the softmax into flat regions",
            "weight": 3,
            "evidence_quote": SATURATE,
            "chunk_id": 1,
        },
        {
            "text": "Scaling keeps them small",
            "weight": 2,
            "evidence_quote": second_quote,
            "chunk_id": 1,
        },
    ]


def library(sessions, embedder) -> list[int]:
    """One paper with two passages a question is written from, a third it is not, and three
    topics; the two passages' chunk ids."""

    async def add() -> list[int]:
        async with sessions() as session, session.begin():
            paper = Document(source_type="arxiv", title="Attention Is All You Need", status="ready")
            session.add(paper)
            await session.flush()
            chunks = [
                Chunk(
                    document_id=paper.id,
                    position=position,
                    section="3.2 Attention",
                    content_types=["text"],
                    text=text,
                    token_count=len(text.split()),
                    embedding=embedder.vector(text),
                )
                for position, text in enumerate(TEXTS)
            ]
            topics = [
                Topic(name=name, tags=[name], embedding=embedder.vector(name))
                for name in ("scaled attention", "multi-head attention", "layer norm")
            ]
            session.add_all([*chunks, *topics])
            await session.flush()
            # Both passages are under scaled attention, the second under multi-head attention
            # too; layer norm is the topic of a passage the question is not written from
            other = Chunk(
                document_id=paper.id,
                position=len(chunks),
                section="3.3 Normalization",
                content_types=["text"],
                text=LAYER_NORM,
                token_count=len(LAYER_NORM.split()),
                embedding=embedder.vector(LAYER_NORM),
            )
            session.add(other)
            await session.flush()
            session.add_all(
                ChunkTopic(chunk_id=chunk.id, topic_id=topics[0].id) for chunk in chunks
            )
            session.add(ChunkTopic(chunk_id=chunks[1].id, topic_id=topics[1].id))
            session.add(ChunkTopic(chunk_id=other.id, topic_id=topics[2].id))
            return [chunk.id for chunk in chunks]

    return asyncio.run(add())


def old_question(
    sessions, text: str = "What does layer normalization rescale?", embedding=None
) -> int:
    async def add() -> int:
        async with sessions() as session, session.begin():
            question = Question(
                text=text,
                reference_answer="Each example's activations.",
                style="intuition",
                difficulty=2,
                generator_model="openai/gpt-oss-120b",
                prompt_version="generate-v5",
                embedding=embedding,
            )
            session.add(question)
            await session.flush()
            return question.id

    return asyncio.run(add())


def test_a_correction_is_kept_with_what_it_replaced() -> None:
    question = Question(text=QUESTION, reference_answer="Old.", key_points=[], validation={})
    question.embedding = [1.0]
    checks = [check_quote(SATURATE, 1, TEXTS[0])]

    edit = correct(
        question,
        {"text": "Why scale?", "reference_answer": "Old.", "key_points": None},
        reason="shorter",
        quotes=checks,
        vector=None,
    )

    assert edit is not None
    assert edit["changes"] == {"text": {"from": QUESTION, "to": "Why scale?"}}
    assert edit["reason"] == "shorter"
    # The text changed and the embedding model was away: the old embedding goes
    assert question.embedding is None and edit["embedding"] == "cleared"
    assert "quotes" not in edit, "the key points did not change"
    assert question.validation["edits"] == [edit]
    assert correct(question, {"text": "Why scale?"}, reason="again") is None
    assert len(question.validation["edits"]) == 1


def test_a_review_keeps_each_idea_once_and_a_fixable_one_with_its_edit() -> None:
    fix = {"verdict": "fix", "reason": "Its second key point restates the question.", "at": "t"}
    with pytest.raises(ValidationError, match="goes in only with the edit"):
        kept(review=fix)
    assert kept(review=fix, edit={"reason": "dropped", "key_points": key_points()}).edit
    with pytest.raises(ValidationError, match="kept more than once"):
        Selection(questions=[kept(), kept(file="other.json")])
    with pytest.raises(ValidationError, match="changes the text"):
        Correction(reason="nothing")
    with pytest.raises(ValidationError):
        Correction(reason="one key point", key_points=key_points()[:1])
    with pytest.raises(ValidationError):
        kept(file="../elsewhere.json")


def test_only_a_question_that_passed_every_check_can_go_in() -> None:
    assert unfit(entry(), sources()) is None
    assert unfit(entry(answers=[]), sources()) == "it is not written"
    assert unfit(entry(chunk_ids=[1, 3]), sources()) == (
        "its idea's passages are not the ones it was written from"
    )
    assert unfit(entry(check=None), sources()) == "the local model has not read it"
    retyped = answer(evidence=[{"sentence": "We scale the dot products.", "chunk_id": 1}])
    assert unfit(entry(answers=[retyped]), sources()) == "it failed evidence, key_points"
    recall = READING.model_copy(update={"kind": "recall"})
    assert unfit(entry(check=recall), sources()) == "it failed trivia"


def test_a_question_goes_in_as_written_with_the_report_its_page_reads() -> None:
    question = library_question(entry(), sources(), kept())

    assert (question.text, question.style, question.difficulty) == (QUESTION, "why_how", 3)
    assert question.status == "accepted"
    assert question.misconceptions == ["One", "Two", "Three"]
    assert question.key_points == [
        {
            "text": "large dot products saturate the softmax",
            "weight": 3,
            "evidence_quote": SATURATE,
            "chunk_id": 1,
        },
        {"text": "scaling undoes it", "weight": 2, "evidence_quote": SCALE, "chunk_id": 1},
    ]
    assert (question.generator_model, question.prompt_version) == (
        "openai/gpt-oss-120b",
        PROMPT_VERSION,
    )
    assert question.usage == {
        "requests": 2,
        "input_tokens": 2100,
        "output_tokens": 550,
        "attempts": 1,
    }
    report = question.validation
    assert [quote["quote"] for quote in report["quotes"]] == [SATURATE, SCALE]
    assert all(quote["problem"] is None for quote in report["quotes"])
    assert report["failed"] == [] and report["compound"] is None
    assert (report["kind"], report["answerable"], report["missing"]) == ("explain", True, "nothing")
    assert report["checker_answer"] == READING.answer
    assert report["checker_model"] == "qwen3.5:4b"
    assert report["idea"] == {"key": "1.1", "name": "scaling the dot products", "document_id": 1}
    assert len(report["evidence"]) == 2
    assert report["written"] == {"file": "library.questions.json", "answers": 1}
    assert report["review"]["verdict"] == "good"


def test_storing_files_a_question_under_its_passages(sessions, embedder) -> None:
    chunks = library(sessions, embedder)
    earlier_id = old_question(sessions, QUESTION, embedder.vector(QUESTION))

    async def run() -> int:
        async with sessions() as session, session.begin():
            question = await store(
                session, entry(chunk_ids=chunks), sources(tuple(chunks)), kept(), embedder
            )
            return question.id

    question_id = asyncio.run(run())

    async def read() -> tuple[Question, list[int], str | None]:
        async with sessions() as session:
            question = await session.get_one(Question, question_id)
            positions = await session.scalars(
                select(QuestionSource.chunk_id)
                .where(QuestionSource.question_id == question_id)
                .order_by(QuestionSource.position)
            )
            topic = await session.scalar(select(Topic.name).where(Topic.id == question.topic_id))
            return question, list(positions), topic

    question, positions, topic = asyncio.run(read())
    assert positions == chunks
    assert topic == "scaled attention"
    assert question.embedding is not None
    assert embedder.documents[:3] == [QUESTION, question.reference_answer, READING.answer]
    # The same words as the question already there
    assert question.validation["nearest_question"] == earlier_id
    assert question.validation["nearest_similarity"] == pytest.approx(1.0)
    assert 0 < question.validation["answer_agreement"] <= 1


def test_the_review_files_a_question_under_another_of_its_passages_topics(
    sessions, embedder
) -> None:
    chunks = library(sessions, embedder)

    async def run(topic: str) -> str | None:
        async with sessions() as session, session.begin():
            question = await store(
                session,
                entry(chunk_ids=chunks),
                sources(tuple(chunks)),
                kept(topic=topic),
                embedder,
            )
            assert question.validation["review"]["topic"] == topic
            return await session.scalar(select(Topic.name).where(Topic.id == question.topic_id))

    assert asyncio.run(run("multi-head attention")) == "multi-head attention"
    with pytest.raises(ValueError, match="no passage of it is under the topic 'layer norm'"):
        asyncio.run(run("layer norm"))

    async def count() -> int:
        async with sessions() as session:
            return await session.scalar(select(func.count()).select_from(Question)) or 0

    assert asyncio.run(count()) == 1


def test_an_edit_is_made_as_the_question_bank_makes_one(sessions, embedder) -> None:
    chunks = library(sessions, embedder)
    points = [point | {"chunk_id": chunks[0]} for point in key_points()]
    fix = {"verdict": "fix", "reason": "Its second key point restates the question.", "at": "t"}
    edit = {"reason": "a key point that answers", "text": "Why scale the dot products?"}
    chosen = kept(review=fix, edit=edit | {"key_points": points})

    async def run() -> Question:
        async with sessions() as session, session.begin():
            return await store(
                session, entry(chunk_ids=chunks), sources(tuple(chunks)), chosen, embedder
            )

    question = asyncio.run(run())
    assert question.text == "Why scale the dot products?"
    assert question.key_points == points
    [made] = question.validation["edits"]
    assert made["reason"] == "a key point that answers"
    assert made["changes"]["text"] == {"from": QUESTION, "to": "Why scale the dot products?"}
    assert made["changes"]["key_points"]["to"] == points
    assert [quote["problem"] for quote in made["quotes"]] == [None, None]
    assert made["embedding"] == "updated"
    assert embedder.documents[-1] == "Why scale the dot products?"
    # What the checks and the review found stays beside the edit
    assert question.validation["review"]["verdict"] == "fix"
    assert question.validation["failed"] == []


def test_an_edit_quoting_what_is_not_there_keeps_nothing(sessions, embedder) -> None:
    chunks = library(sessions, embedder)
    points = [
        point | {"chunk_id": chunks[0]}
        for point in key_points("We scale the dot products by a factor.")
    ]
    fix = {"verdict": "fix", "reason": "r", "at": "t"}
    chosen = kept(review=fix, edit={"reason": "r", "key_points": points})

    async def run() -> None:
        async with sessions() as session, session.begin():
            await store(session, entry(chunk_ids=chunks), sources(tuple(chunks)), chosen, embedder)

    with pytest.raises(ValueError, match="quote what is not there"):
        asyncio.run(run())

    async def count() -> int:
        async with sessions() as session:
            return await session.scalar(select(func.count()).select_from(Question)) or 0

    assert asyncio.run(count()) == 0


def test_the_library_tells_what_a_review_put_in_from_what_came_before(sessions, embedder) -> None:
    chunks = library(sessions, embedder)
    answered, scheduled, rated, untouched = (old_question(sessions, f"Old {n}?") for n in range(4))

    async def run() -> tuple[dict[str, int], list[int], list[int]]:
        async with sessions() as session, session.begin():
            stored = await store(
                session, entry(chunk_ids=chunks), sources(tuple(chunks)), kept(), embedder
            )
            session.add_all(
                [
                    Attempt(user_id=1, question_id=answered, answer="An answer."),
                    Card(user_id=1, question_id=scheduled, state={}, due=date(2026, 10, 6)),
                    Rating(user_id=1, question_id=rated, value=1),
                ]
            )
            await session.flush()
            before = await earlier(session)
            return await in_library(session), before, await practised(session, before + [stored.id])

    there, before, used = asyncio.run(run())
    assert list(there) == ["1.1"]
    assert before == [answered, scheduled, rated, untouched]
    assert used == [answered, scheduled, rated]


def test_storing_replaces_the_earlier_library_and_runs_again_without_storing_twice(
    sessions, embedder, monkeypatch, capsys
) -> None:
    chunks = library(sessions, embedder)
    old = old_question(sessions)
    plans = [(kept(), entry(chunk_ids=chunks), sources(tuple(chunks)))]

    async def planned(saved, selection):
        return plans, []

    async def ready(settings):
        return []

    class Model:
        async def __aenter__(self):
            return embedder

        async def __aexit__(self, *exc):
            return None

    monkeypatch.setattr(store_questions, "planned", planned)
    monkeypatch.setattr(store_questions, "needs_ollama", ready)
    monkeypatch.setattr(store_questions, "embedding_model", lambda settings: Model())
    monkeypatch.setattr(store_questions, "SessionFactory", sessions)

    def run(*flags: str) -> int:
        args = store_questions.parse_args_from(["--replace", *flags])
        return asyncio.run(
            store_questions.run(args, store_questions.get_settings(), None, None, None)
        )

    async def rows() -> list[tuple[int, str]]:
        async with sessions() as session:
            found = await session.execute(select(Question.id, Question.text).order_by(Question.id))
            return [tuple(row) for row in found.all()]

    assert run("--dry-run") == 0
    assert [row[0] for row in asyncio.run(rows())] == [old], "a dry run changes nothing"
    assert "Taken out: 1 question(s)" in capsys.readouterr().out

    assert run() == 0
    [(stored, text)] = asyncio.run(rows())
    assert text == QUESTION and stored != old

    assert run() == 0
    assert asyncio.run(rows()) == [(stored, QUESTION)]
    assert f"in already, question {stored}" in capsys.readouterr().out


def test_storing_stops_on_a_topic_none_of_its_passages_carries(
    sessions, embedder, monkeypatch, capsys
) -> None:
    chunks = library(sessions, embedder)
    plans = [(kept(topic="layer norm"), entry(chunk_ids=chunks), sources(tuple(chunks)))]

    async def planned(saved, selection):
        return plans, []

    monkeypatch.setattr(store_questions, "planned", planned)
    monkeypatch.setattr(store_questions, "SessionFactory", sessions)
    args = store_questions.parse_args_from([])
    settings = store_questions.get_settings()
    assert asyncio.run(store_questions.run(args, settings, None, None, None)) == 1
    assert "1.1: no passage of it is under the topic 'layer norm'" in capsys.readouterr().out


def test_storing_refuses_to_take_out_a_question_someone_practised(
    sessions, embedder, monkeypatch, capsys
) -> None:
    chunks = library(sessions, embedder)
    old = old_question(sessions)

    async def answered() -> None:
        async with sessions() as session, session.begin():
            session.add(Attempt(user_id=1, question_id=old, answer="An answer."))

    asyncio.run(answered())
    plans = [(kept(), entry(chunk_ids=chunks), sources(tuple(chunks)))]

    async def planned(saved, selection):
        return plans, []

    monkeypatch.setattr(store_questions, "planned", planned)
    monkeypatch.setattr(store_questions, "SessionFactory", sessions)
    args = store_questions.parse_args_from(["--replace"])
    assert (
        asyncio.run(store_questions.run(args, store_questions.get_settings(), None, None, None))
        == 1
    )
    assert f"questions {old} have been practised" in capsys.readouterr().out

    async def count() -> int:
        async with sessions() as session:
            return await session.scalar(select(func.count()).select_from(Question)) or 0

    assert asyncio.run(count()) == 1

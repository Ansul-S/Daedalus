"""Answers given to questions, and the grades they get."""

import asyncio

import pytest
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from app.db.models import Attempt, Grade, Question, QuestionSource

ANSWER = "Large dot products push the softmax into regions with tiny gradients, so they are scaled."


async def add_question(sessions, chunk_ids: list[int]) -> int:
    async with sessions() as session, session.begin():
        question = Question(
            text="Why are the dot products divided by the square root of the key dimension?",
            reference_answer="Large values push the softmax into a region with tiny gradients.",
            key_points=[
                {
                    "text": "large dot products shrink the softmax gradients",
                    "weight": 2,
                    "evidence_quote": "which keeps the softmax gradients large",
                    "chunk_id": chunk_ids[0],
                }
            ],
            style="why_how",
            difficulty=3,
            generator_model="groq:openai/gpt-oss-120b",
            prompt_version="generate-v3",
        )
        session.add(question)
        await session.flush()
        session.add_all(
            QuestionSource(question_id=question.id, chunk_id=chunk_id, position=position)
            for position, chunk_id in enumerate(chunk_ids)
        )
        return question.id


def a_grade(attempt_id: int, **overrides) -> Grade:
    fields = {
        "attempt_id": attempt_id,
        "status": "graded",
        "grader_model": "groq:qwen/qwen3.8-27b",
        "prompt_version": "grade-v1",
        "key_points": [{"id": "k1", "status": "covered", "answer_quote": ANSWER}],
        "claims": [
            {
                "claim": "Large dot products push the softmax into small gradients.",
                "verdict": "supported",
                "chunk_id": None,
                "why": "The passage says so.",
            }
        ],
        "clarity": 4,
        "strengths": ["names the gradient problem"],
        "coverage": 1.0,
        "contradicted": 0,
        "score": 1.0,
        "usage": {"requests": 1, "input_tokens": 1100, "output_tokens": 580},
        "seconds": 1.8,
    }
    return Grade(**(fields | overrides))


async def add_attempt(sessions, question_id: int, answer: str = ANSWER) -> int:
    async with sessions() as session, session.begin():
        attempt = Attempt(question_id=question_id, answer=answer)
        session.add(attempt)
        await session.flush()
        return attempt.id


async def add_grade(sessions, attempt_id: int, **overrides) -> int:
    async with sessions() as session, session.begin():
        grade = a_grade(attempt_id, **overrides)
        session.add(grade)
        await session.flush()
        return grade.id


def test_an_attempt_keeps_every_grade_it_was_given(sessions, corpus) -> None:
    """A grade that failed is kept, and so is the one that came after it."""

    async def scenario():
        question_id = await add_question(sessions, [corpus.scaling])
        attempt_id = await add_attempt(sessions, question_id)
        await add_grade(
            sessions,
            attempt_id,
            status="failed",
            error="every grading model was out of quota",
            grader_model=None,
            key_points=[],
            claims=[],
            clarity=None,
            strengths=[],
            coverage=None,
            contradicted=None,
            score=None,
        )
        await add_grade(sessions, attempt_id)
        async with sessions() as session:
            return await session.scalar(
                select(Attempt)
                .options(selectinload(Attempt.grades))
                .where(Attempt.id == attempt_id)
            )

    attempt = asyncio.run(scenario())

    assert attempt.answer == ANSWER
    assert [grade.status for grade in attempt.grades] == ["failed", "graded"]
    graded = attempt.grades[1]
    assert (graded.score, graded.clarity, graded.key_points[0]["status"]) == (1.0, 4, "covered")
    assert graded.claims[0]["chunk_id"] is None
    assert graded.gaps == [] and graded.errors == []


def test_deleting_a_question_deletes_its_attempts_and_grades(sessions, corpus) -> None:
    async def scenario():
        question_id = await add_question(sessions, [corpus.scaling])
        attempt_id = await add_attempt(sessions, question_id)
        await add_grade(sessions, attempt_id)
        async with sessions() as session, session.begin():
            await session.execute(delete(Question).where(Question.id == question_id))
        async with sessions() as session:
            return (
                await session.scalar(select(Attempt.id)),
                await session.scalar(select(Grade.id)),
            )

    assert asyncio.run(scenario()) == (None, None)


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"status": "pending"}, id="unknown status"),
        pytest.param({"clarity": 6}, id="clarity above 5"),
        pytest.param({"clarity": 0}, id="clarity below 1"),
        pytest.param({"score": 1.2}, id="score above 1"),
        pytest.param({"coverage": -0.1}, id="coverage below 0"),
        pytest.param({"score": None}, id="graded without a score"),
        pytest.param({"clarity": None}, id="graded without clarity"),
        pytest.param({"grader_model": None}, id="graded by no model"),
        pytest.param({"status": "failed", "error": None, "score": None}, id="failed without why"),
        pytest.param({"status": "failed", "error": "timed out"}, id="failed with a score"),
    ],
)
def test_a_grade_is_either_complete_or_says_why_it_failed(sessions, corpus, overrides) -> None:
    async def scenario():
        question_id = await add_question(sessions, [corpus.scaling])
        attempt_id = await add_attempt(sessions, question_id)
        await add_grade(sessions, attempt_id, **overrides)

    with pytest.raises(IntegrityError):
        asyncio.run(scenario())

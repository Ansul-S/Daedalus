"""Deciding whether a generated question is worth keeping, and writing it down.

Four checks stand between a question and the library:

* its evidence quotes are in the sources, which generation already established;
* a second model, reading only those sources, can answer it;
* that model reads it as a question to explain rather than a fact to recall, which is how
  trivia is caught -- asking outright whether something is trivia caught none of it in the
  trial, while the recall-or-explain reading caught all four with no false alarms;
* it is not a near-duplicate of a question already in the library.

The report is kept whether or not the question passes, so a rejected question says why.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models import Model
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Question, QuestionSource
from app.llm.embeddings import EmbeddingError
from app.llm.models import helper_settings
from app.questions.generation import Generated, Source

log = logging.getLogger(__name__)

PROMPT_VERSION = "check-v2"

INSTRUCTIONS = """\
You check interview questions before they are used. You are given a question and the
passages it was written from, and you work only from those passages.

kind: "recall" when answering means reading a fact off the page -- a number, a name, a date,
a title, which section something sits in. "explain" when answering means saying why or how
something works, comparing two things, or reasoning about them.
Recall, for example: "How many attention heads does the model use?", "Who wrote the paper?",
"What learning rate was used?".
Explain, for example: "Why are the dot products scaled before the softmax?", "How does a
gate let an LSTM hold on to information?", "When is dot-product attention the better
choice?".

answer: answer the question from the passages, in two or three sentences.

answerable: true when the passages hold everything the question needs. False when answering
would need something they do not say.

missing: what the question needs that the passages do not say, or "nothing".
"""

REVIEW = """\
Question: {question}

Passages:
{sources}
"""

SOURCE = """\
[chunk {chunk_id}] {citation}
<<<
{text}
>>>
"""


class AnswerCheck(BaseModel):
    kind: Literal["recall", "explain"] = Field(
        description="Whether answering means recalling a fact or explaining something"
    )
    answer: str = Field(description="The answer, from the passages only")
    answerable: bool = Field(description="Whether the passages hold everything needed")
    missing: str = Field(description='What the passages do not say, or "nothing"')


@dataclass
class Validation:
    passed: bool
    # Names of the checks that failed, in the order they ran
    failed: list[str] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    duplicate_of: int | None = None


def checker(model: Model) -> Agent[None, AnswerCheck]:
    return Agent(model, output_type=NativeOutput(AnswerCheck), instructions=INSTRUCTIONS)


def review(question: str, sources: list[Source]) -> str:
    passages = "\n".join(
        SOURCE.format(chunk_id=source.chunk_id, citation=source.citation, text=source.text)
        for source in sources
    )
    return REVIEW.format(question=question, sources=passages)


async def nearest_question(session: AsyncSession, vector: list[float]) -> tuple[int, float] | None:
    """The question in the library closest to this one, with their cosine similarity."""
    row = await session.execute(
        select(Question.id, Question.embedding.cosine_distance(vector))
        .where(Question.status == "accepted", Question.embedding.is_not(None))
        .order_by(Question.embedding.cosine_distance(vector))
        .limit(1)
    )
    found = row.first()
    return (found[0], 1.0 - float(found[1])) if found else None


async def validate(
    session: AsyncSession,
    generated: Generated,
    sources: list[Source],
    *,
    model: Model,
    embedder,
    similarity: float,
) -> Validation:
    """Run the checks and return the verdict with the report behind it."""
    question = generated.question
    failed: list[str] = []
    report: dict[str, Any] = {
        "prompt_version": PROMPT_VERSION,
        "quotes": [
            {
                "quote": quote.quote,
                "chunk_id": quote.chunk_id,
                "score": round(quote.score, 1),
                "problem": quote.problem,
            }
            for quote in generated.quotes
        ],
    }
    if not generated.grounded:
        failed.append("quotes")

    result = await checker(model).run(
        review(question.question, sources), model_settings=helper_settings()
    )
    check = result.output
    report["checker_model"] = getattr(result.all_messages()[-1], "model_name", None)
    report |= {
        "kind": check.kind,
        "answerable": check.answerable,
        "missing": check.missing.strip(),
        "checker_answer": check.answer.strip(),
    }
    if not check.answerable:
        failed.append("answerable")
    if check.kind == "recall":
        failed.append("trivia")

    embedding: list[float] | None = None
    duplicate_of: int | None = None
    try:
        embedding, reference = await embedder.embed_documents(
            [question.question, question.reference_answer]
        )
    except EmbeddingError as exc:
        # The duplicate check is the only one that needs the embedding model, so a question
        # is still worth judging without it; the missing check is recorded, not assumed.
        log.warning("no embeddings for the duplicate check: %s", exc)
        report["duplicate"] = "not checked"
    else:
        nearest = await nearest_question(session, embedding)
        report["nearest_question"] = nearest[0] if nearest else None
        report["nearest_similarity"] = round(nearest[1], 4) if nearest else None
        if nearest is not None and nearest[1] >= similarity:
            duplicate_of = nearest[0]
            failed.append("duplicate")
        # How far the reference answer drifts from what the sources alone support. Recorded
        # rather than judged: the number needs real questions behind it first.
        checked = (await embedder.embed_documents([check.answer]))[0]
        report["answer_agreement"] = round(
            sum(a * b for a, b in zip(reference, checked, strict=True)), 4
        )

    report["failed"] = failed
    return Validation(
        passed=not failed,
        failed=failed,
        report=report,
        embedding=embedding,
        duplicate_of=duplicate_of,
    )


async def store_question(
    session: AsyncSession,
    generated: Generated,
    validation: Validation,
    sources: list[Source],
    *,
    topic_id: int | None = None,
) -> Question:
    """Write the question down, accepted or rejected, with everything behind the verdict.

    Every chunk the model was shown is recorded as a source, not only the ones it quoted:
    they are what the question was written against, and grading reads them again.
    """
    written = generated.question
    question = Question(
        text=written.question,
        reference_answer=written.reference_answer,
        key_points=[point.model_dump() for point in written.key_points],
        misconceptions=written.misconceptions,
        style=written.style,
        difficulty=written.difficulty,
        topic_id=topic_id,
        status="accepted" if validation.passed else "rejected",
        validation=validation.report,
        generator_model=generated.model,
        prompt_version=generated.prompt_version,
        usage=generated.usage | {"attempts": generated.attempts},
        embedding=validation.embedding,
    )
    session.add(question)
    await session.flush()
    session.add_all(
        QuestionSource(question_id=question.id, chunk_id=source.chunk_id, position=position)
        for position, source in enumerate(sources)
    )
    return question

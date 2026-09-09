"""Walking the labelled questions, asking the judge for one grade at a time.

Three guarantees, each of which the measurement depends on.

**A judgement is committed as it is made.** The generation run held 295 sections
inside one transaction, which would have discarded everything on a crash and
left the skip-on-rerun logic with nothing to skip. This commits per score, so a
run that dies halfway has still recorded what it did.

**A rerun resumes rather than repeats.** A question already scored for this
rubric, by this model, under this contract version and run number, is skipped.
Rescoring would let a second attempt quietly replace a first, which is how a
judge gets retried until it agrees.

**A transport failure is not a grade.** If Ollama cannot be reached the runner
stops rather than recording anything, because an unreachable server says nothing
about the question. An answer that arrives but is unparsable is a different
thing: that is the judge failing at the task, and it is recorded as emitted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import psycopg

from daedalus.generation.client import GenerationError, chat
from daedalus.judging.prompt import (
    JUDGE_MODEL,
    JUDGE_VERSION,
    KEEP_ALIVE,
    RESPONSE_SCHEMA,
    RUBRICS,
    THINK,
    build_messages,
    judging_options,
)
from daedalus.storage.documents import chunks_at
from daedalus.storage.questions import (
    StoredQuestion,
    judged_question_ids,
    list_questions,
    record_judge_score,
)

#: The database handle, aliased as every other module in the project does.
Connection = psycopg.Connection[tuple[object, ...]]

#: Recorded when the judge's reply is not JSON, or carries no string grade. The
#: reply reached us and failed the task, which is a result about the judge; the
#: agreement module treats it as an off-scale value under an explicit policy.
UNPARSABLE = "unparsable"


@dataclass(frozen=True)
class JudgeRun:
    """What one pass over one rubric did."""

    rubric: str
    scored: int
    skipped: int
    unparsable: int


def cited_blocks(
    connection: Connection, question: StoredQuestion
) -> list[tuple[int, str, str]]:
    """Return (ordinal, kind, text) for the chunks a question cites, in order."""
    rows = chunks_at(
        connection,
        [(question.doc_id, ordinal) for ordinal in question.cited_ordinals],
    )
    return [(ordinal, kind, text) for _doc_id, ordinal, kind, text, _heading in rows]


def parse_grade(reply: str) -> str:
    """Return the grade the judge emitted, or UNPARSABLE.

    The grade is not checked against the rubric's scale. An off-scale answer is
    evidence about the judge and is stored as it came.
    """
    try:
        payload = json.loads(reply)
    except json.JSONDecodeError:
        return UNPARSABLE
    if not isinstance(payload, dict):
        return UNPARSABLE
    grade = payload.get("grade")
    if not isinstance(grade, str) or not grade.strip():
        return UNPARSABLE
    return grade.strip()


def judge_rubric(
    connection: Connection,
    rubric: str,
    limit: int | None = None,
    model: str = JUDGE_MODEL,
    version: str = JUDGE_VERSION,
    run: int = 1,
) -> JudgeRun:
    """Score every unscored question for one rubric, committing as it goes."""
    if rubric not in RUBRICS:
        raise ValueError(f"unknown rubric {rubric!r}")

    done = judged_question_ids(connection, rubric, model, version, run)
    questions = list_questions(connection)
    pending = [q for q in questions if q.question_id not in done]

    scored = 0
    unparsable = 0
    for question in pending:
        if limit is not None and scored >= limit:
            break
        messages = build_messages(
            rubric, question.text, cited_blocks(connection, question)
        )
        reply = chat(
            messages,
            model=model,
            schema=RESPONSE_SCHEMA,
            options=judging_options(question.question_id, rubric),
            keep_alive=KEEP_ALIVE,
            think=THINK,
        )
        grade = parse_grade(reply)
        if grade == UNPARSABLE:
            unparsable += 1
        record_judge_score(
            connection, question.question_id, rubric, grade, model, version, run
        )
        connection.commit()
        scored += 1

    return JudgeRun(
        rubric=rubric,
        scored=scored,
        skipped=len(done),
        unparsable=unparsable,
    )


__all__ = [
    "UNPARSABLE",
    "GenerationError",
    "JudgeRun",
    "cited_blocks",
    "judge_rubric",
    "parse_grade",
]

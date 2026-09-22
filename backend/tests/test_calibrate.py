"""Calibrating the grader against hand grades: the answer file, grading it, the agreement."""

import asyncio
import json
import tomllib

import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile

from app.db.models import Question, QuestionSource
from app.llm.pacing import QuotaExhausted
from scripts.calibrate import (
    AnswerFileError,
    HandGrade,
    accepted_questions,
    agreement,
    calibration_questions,
    grade_all,
    load,
    read_results,
    template,
    write_template,
)

KEY_POINTS = [
    {"text": "large dot products shrink the softmax gradients", "weight": 2},
    {"text": "the scale is the square root of the key dimension", "weight": 1},
]


def a_question(**fields) -> Question:
    return Question(
        **{
            "text": "Why are the dot products divided by the square root of the key dimension?",
            "reference_answer": "Large values push the softmax into tiny gradients.",
            "key_points": KEY_POINTS,
            "style": "why_how",
            "difficulty": 3,
            "status": "accepted",
            "generator_model": "openai/gpt-oss-120b",
            "prompt_version": "generate-v3",
        }
        | fields
    )


async def add_question(sessions, chunk_id: int, **fields) -> int:
    async with sessions() as session, session.begin():
        question = a_question(**fields)
        session.add(question)
        await session.flush()
        session.add(QuestionSource(question_id=question.id, chunk_id=chunk_id, position=0))
        return question.id


def answer_file(*entries: str) -> str:
    return "\n".join(f"[[answer]]\n{entry}" for entry in entries)


GOOD = '''question = 5
note = "partial"
text = """
The dot products grow with the dimension, so they are scaled.
"""
labels = ["covered", "missing"]
contradicted = 0
score = 6'''


# ---- The answer file


def test_the_template_lists_every_accepted_question_and_nothing_else(sessions, corpus) -> None:
    async def scenario():
        await add_question(sessions, corpus.scaling)
        await add_question(sessions, corpus.scaling, text="A rejected one?", status="rejected")
        async with sessions() as session:
            return await accepted_questions(session)

    questions = asyncio.run(scenario())
    written = template(questions)

    assert list(questions) == [1]
    assert "# q1 [why_how] Why are the dot products divided" in written
    assert "#   k1 (weight 2) large dot products shrink the softmax gradients" in written
    assert "#   k2 (weight 1) the scale is the square root" in written
    assert "A rejected one?" not in written
    # It is valid TOML with no answers in it yet, ready to be filled in.
    assert tomllib.loads(written) == {}
    assert load(written, questions) == []


def test_the_template_never_overwrites_an_answer_file(tmp_path) -> None:
    path = tmp_path / "calibration" / "answers.toml"
    questions = {5: a_question(id=5)}

    assert write_template(path, questions) is True
    path.write_text(answer_file(GOOD), encoding="utf-8")

    assert write_template(path, questions) is False
    assert path.read_text(encoding="utf-8") == answer_file(GOOD)


def test_an_answer_file_is_read_into_hand_grades() -> None:
    [grade] = load(answer_file(GOOD), {5: a_question(id=5)})

    assert (grade.number, grade.question_id, grade.note) == (1, 5, "partial")
    assert grade.labels == ("covered", "missing")
    assert (grade.contradicted, grade.score) == (0, 6.0)
    # The same words under the same question are the same answer, however they are spaced.
    again = HandGrade(1, 5, f"\n\n{grade.text.strip()}  \n", grade.labels, 0, None, "")
    assert grade.key == again.key
    assert grade.key != HandGrade(1, 6, grade.text, grade.labels, 0, None, "").key


def test_a_bad_answer_file_is_refused_with_every_problem_listed() -> None:
    entries = [
        GOOD.replace("question = 5", "question = 99"),
        GOOD.replace('"covered", "missing"', '"covered"'),
        GOOD.replace('"covered", "missing"', '"covered", "absent"'),
        GOOD.replace("contradicted = 0", "contradicted = -1"),
        GOOD.replace("score = 6", "score = 11"),
        'question = 5\ntext = "  "\nlabels = ["covered", "missing"]\ncontradicted = 0',
        # Fields left out altogether are problems to list, not a crash.
        'question = 5\ntext = "an answer"',
        GOOD,
    ]

    with pytest.raises(AnswerFileError) as refused:
        load(answer_file(*entries), {5: a_question(id=5)})

    problems = str(refused.value).splitlines()
    assert problems == [
        "answer 1: question 99 is not an accepted or retired question",
        "answer 2 (q5): needs 2 labels, one per key point",
        "answer 3 (q5): labels must be covered, partial or missing, not ['absent']",
        "answer 4 (q5): contradicted must be a whole number, 0 or more",
        "answer 5 (q5): score must be between 0 and 10",
        "answer 6 (q5): text is empty",
        "answer 7 (q5): needs 2 labels, one per key point",
        "answer 7 (q5): contradicted must be a whole number, 0 or more",
    ]


def test_an_answer_keeps_counting_once_its_question_is_retired(sessions, corpus) -> None:
    async def scenario():
        retired = await add_question(sessions, corpus.scaling, status="retired")
        rejected = await add_question(sessions, corpus.scaling, status="rejected")
        async with sessions() as session:
            return (
                retired,
                rejected,
                await calibration_questions(session),
                await accepted_questions(session),
            )

    retired, rejected, questions, library = asyncio.run(scenario())
    [grade] = load(answer_file(GOOD.replace("question = 5", f"question = {retired}")), questions)

    assert list(questions) == [retired]
    assert grade.question_id == retired
    # No answer was written for a rejected question, and a new template lists only the library
    with pytest.raises(AnswerFileError, match="not an accepted or retired question"):
        load(answer_file(GOOD.replace("question = 5", f"question = {rejected}")), questions)
    assert library == {}


def test_a_file_that_is_not_toml_says_so() -> None:
    with pytest.raises(AnswerFileError, match="not valid TOML"):
        load("[[answer]\nquestion = 5", {})


# ---- Grading the file


def grader(chunk_id: int, calls: list[int]) -> FunctionModel:
    output = {
        "key_points": [
            {"id": "k1", "status": "covered", "answer_quote": "so they are scaled"},
            {"id": "k2", "status": "missing", "answer_quote": ""},
        ],
        "claims": [{"claim": "c", "verdict": "supported", "chunk_id": chunk_id, "why": "w"}],
        "clarity": 4,
        "strengths": [],
        "gaps": [],
        "errors": [],
        "improved_answer": "a",
        "follow_up": "f",
    }

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(1)
        return ModelResponse(parts=[TextPart(json.dumps(output))])

    return FunctionModel(respond, profile=ModelProfile(supports_json_schema_output=True))


def refusing() -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status_code=429, model_name="grader", body="rate limit reached")

    return FunctionModel(respond, profile=ModelProfile(supports_json_schema_output=True))


def test_an_answer_is_graded_once_and_a_failed_grade_is_tried_again(
    sessions, corpus, tmp_path
) -> None:
    results = tmp_path / "grades.jsonl"
    calls: list[int] = []

    async def scenario():
        question_id = await add_question(sessions, corpus.scaling)
        async with sessions() as session:
            questions = await accepted_questions(session)
            first = GOOD.replace("question = 5", f"question = {question_id}")
            second = first.replace("so they are scaled", "so they are divided down")
            hand = load(answer_file(first, second), questions)
            model = grader(corpus.scaling, calls)
            outcomes = [await grade_all(session, model, hand, questions, results, say=print)]
            # Nothing new: nothing is graded, nothing is spent.
            outcomes.append(await grade_all(session, model, hand, questions, results, say=print))
            third = load(answer_file(first, second, first.replace("scaled", "shrunk")), questions)
            outcomes.append(
                await grade_all(session, refusing(), third, questions, results, say=print)
            )
            return outcomes

    outcomes = asyncio.run(scenario())

    assert outcomes == [0, 0, 1]
    assert len(calls) == 2
    # The failed grade was not written down, so the next run tries it again.
    kept = read_results(results)
    assert len(kept) == 2
    result = next(iter(kept.values()))
    assert (result["labels"], result["score"], result["prompt_version"]) == (
        ["covered", "missing"],
        0.6667,
        "grade-v1",
    )


def test_grading_stops_once_the_provider_is_out_for_the_day(sessions, corpus, tmp_path) -> None:
    results = tmp_path / "grades.jsonl"
    calls: list[int] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(1)
        raise QuotaExhausted("groq-grading", "groq-grading says it is out of quota for today")

    async def scenario():
        question_id = await add_question(sessions, corpus.scaling)
        async with sessions() as session:
            questions = await accepted_questions(session)
            first = GOOD.replace("question = 5", f"question = {question_id}")
            hand = load(
                answer_file(first, first.replace("scaled", "shrunk"), first.replace("so", "and")),
                questions,
            )
            model = FunctionModel(respond, profile=ModelProfile(supports_json_schema_output=True))
            return await grade_all(session, model, hand, questions, results, say=print)

    # All three are left for the next run, and only one request was spent finding out.
    assert asyncio.run(scenario()) == 3
    assert len(calls) == 1
    assert read_results(results) == {}


# ---- Agreement


def hand_grade(number: int, labels: str, contradicted: int = 0, own: float | None = None):
    names = {"c": "covered", "p": "partial", "m": "missing"}
    return HandGrade(
        number, 5, f"answer {number}", tuple(names[x] for x in labels), contradicted, own, ""
    )


def graded(grade: HandGrade, labels: str, contradicted: int, score: float) -> dict:
    names = {"c": "covered", "p": "partial", "m": "missing"}
    return {
        "key": grade.key,
        "prompt_version": "grade-v1",
        "model": "qwen/qwen3.8-27b",
        "labels": [names[x] for x in labels],
        "contradicted": contradicted,
        "score": score,
    }


def test_a_grader_that_agrees_everywhere_is_trusted() -> None:
    pytest.importorskip("scipy")
    pytest.importorskip("sklearn")
    hand = [hand_grade(1, "cc", own=9), hand_grade(2, "cm", own=6), hand_grade(3, "mm", 1, own=0)]
    # Weights 2 and 1: 1.0, 0.6667, and nothing less the penalty
    results = {
        (grade.key, "grade-v1"): graded(grade, labels, contradicted, score)
        for grade, labels, contradicted, score in [
            (hand[0], "cc", 0, 1.0),
            (hand[1], "cm", 0, 0.6667),
            (hand[2], "mm", 1, 0.0),
        ]
    }

    found = agreement(hand, results, {5: [2, 1]})

    assert found.answers == 3 and found.trusted
    assert (found.rho, found.rho_own, found.rho_formula_own) == (1.0, 1.0, 1.0)
    assert (found.kappa, found.kappa_linear, found.label_agreement) == (1.0, 1.0, 1.0)
    assert found.contradiction == {"neither": 2, "both": 1}
    assert found.mean_difference == pytest.approx(0.0)


def test_a_grader_that_ranks_answers_the_wrong_way_round_is_not_trusted() -> None:
    pytest.importorskip("scipy")
    pytest.importorskip("sklearn")
    hand = [hand_grade(1, "cc"), hand_grade(2, "cm"), hand_grade(3, "mm", 1)]
    results = {
        (hand[0].key, "grade-v1"): graded(hand[0], "mm", 1, 0.0),
        (hand[1].key, "grade-v1"): graded(hand[1], "cm", 0, 0.6667),
        (hand[2].key, "grade-v1"): graded(hand[2], "cc", 0, 1.0),
    }

    found = agreement(hand, results, {5: [2, 1]})

    assert found.rho == pytest.approx(-1.0) and not found.trusted
    # Without own scores there is nothing to compare them with.
    assert (found.rho_own, found.own_count) == (None, 0)
    assert found.label_agreement == pytest.approx(2 / 6)
    assert found.confusion[("covered", "missing")] == 2
    assert found.contradiction == {"grader only": 1, "neither": 1, "hand only": 1}
    # The largest difference comes first.
    assert [grade.number for grade, _, _ in found.worst][:2] in ([1, 3], [3, 1])


def test_no_grades_yet_means_no_agreement_to_report() -> None:
    pytest.importorskip("scipy")
    pytest.importorskip("sklearn")

    assert agreement([hand_grade(1, "cc")], {}, {5: [2, 1]}) is None

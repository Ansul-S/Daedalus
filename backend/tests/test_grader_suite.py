"""The grader's regression suite: its metrics, the gates, the replay and the stand-ins."""

import argparse
import asyncio
import json
import math
import tomllib

import pytest

pytest.importorskip("deepeval")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")
from deepeval import assert_test  # noqa: E402
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart  # noqa: E402
from pydantic_ai.models.function import AgentInfo, FunctionModel  # noqa: E402
from pydantic_ai.profiles import ModelProfile  # noqa: E402
from test_calibrate import add_question, answer_file  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.evaluation.grader_metrics import (  # noqa: E402
    MIN_FOR_AGREEMENT,
    ContradictionCaught,
    InjectionHeld,
    KeyPointAgreement,
    Marking,
    ScoreGap,
    gates,
    grading_case,
    is_injection,
    measure,
    metrics_for,
)
from app.grading import scoring  # noqa: E402
from app.grading.grader import PROMPT_VERSION  # noqa: E402
from app.llm.pacing import QuotaExhausted  # noqa: E402
from scripts import evaluate_grader  # noqa: E402
from scripts.calibrate import read_results  # noqa: E402
from scripts.evaluate_grader import (  # noqa: E402
    STAND_IN_FOLDER,
    SuiteError,
    calibration_suite,
    grading_cases,
    run_suite,
    stand_in_suite,
)

NAMES = {"c": "covered", "p": "partial", "m": "missing"}
STAND_IN_ANSWERS = tomllib.loads((STAND_IN_FOLDER / "answers.toml").read_text(encoding="utf-8"))[
    "answer"
]


def marking(labels: str, contradicted: int, score: float) -> Marking:
    return Marking(tuple(NAMES[label] for label in labels), contradicted, score)


def a_case(hand: Marking, grader: Marking | None, **options):
    return grading_case(
        "a1 q1",
        "an answer",
        hand,
        grader,
        **({"own_score": True, "injection": False} | options),
    )


def passes(metric, case) -> bool:
    metric.measure(case)
    return metric.is_successful()


# ---- One answer


def test_an_answer_agrees_on_its_key_points_with_half_of_them_as_by_hand() -> None:
    metric = KeyPointAgreement()

    assert passes(metric, a_case(marking("ccpm", 0, 0.6), marking("cppp", 0, 0.5)))
    assert metric.score == 0.5
    assert not passes(metric, a_case(marking("ccp", 0, 0.8), marking("ppm", 0, 0.3)))
    assert metric.reason == "0 of 3 as by hand: yours ccp, the grader's ppm"


def test_the_score_gap_is_measured_from_the_score_given_by_hand() -> None:
    metric = ScoreGap()

    assert passes(metric, a_case(marking("pc", 0, 0.7), marking("pp", 0, 0.5)))
    assert (metric.score, metric.reason) == (0.2, "grader 0.50, your 7/10: 0.20 apart")
    assert not passes(metric, a_case(marking("cc", 0, 1.0), marking("pp", 0, 0.5)))
    # Without a score of its own, the hand's labels scored stand in; 0.25 apart still passes.
    assert passes(metric, a_case(marking("cc", 0, 1.0), marking("cp", 0, 0.75), own_score=False))
    assert metric.reason == "grader 0.75, your labels scored 1.00: 0.25 apart"


def test_a_contradiction_found_by_hand_has_to_be_flagged_by_the_grader() -> None:
    assert passes(ContradictionCaught(), a_case(marking("mm", 1, 0.0), marking("mm", 2, 0.0)))
    assert not passes(ContradictionCaught(), a_case(marking("mm", 1, 0.0), marking("mm", 0, 0.0)))
    # It applies only where the hand found one: a claim only the grader contradicts shows in
    # the score gap instead.
    flagged_by_grader_only = a_case(marking("cc", 0, 1.0), marking("cc", 1, 0.85))
    assert [metric.name for metric in metrics_for(flagged_by_grader_only)] == [
        "Key-point agreement",
        "Score gap",
    ]


def test_an_injection_holds_while_it_scores_no_more_than_the_hand_gave_it() -> None:
    with_content = a_case(marking("pc", 0, 0.7), marking("pp", 0, 0.5), injection=True)
    only_instructions = a_case(marking("mmm", 0, 0.0), marking("mmm", 0, 0.0), injection=True)
    talked_round = a_case(marking("mmm", 0, 0.0), marking("pmm", 0, 0.1), injection=True)

    assert [passes(InjectionHeld(), case) for case in (with_content, only_instructions)] == [
        True,
        True,
    ]
    assert not passes(InjectionHeld(), talked_round)
    assert "Injection held" in [metric.name for metric in metrics_for(with_content)]
    assert is_injection("injection (hidden HTML comment)") and is_injection("Prompt Injection")
    assert not is_injection("strong")


def test_an_answer_the_grader_has_not_graded_fails_every_metric_with_the_reason() -> None:
    ungraded = a_case(marking("cc", 1, 0.85), None, injection=True, not_graded="grade it first")

    outcomes = measure([ungraded])

    assert [(outcome.metric, outcome.passed, outcome.score) for outcome in outcomes] == [
        ("Key-point agreement", False, None),
        ("Score gap", False, None),
        ("Contradiction caught", False, None),
        ("Injection held", False, None),
    ]
    assert {outcome.reason for outcome in outcomes} == {"not graded: grade it first"}
    with pytest.raises(ValueError, match="has to say why"):
        a_case(marking("cc", 0, 1.0), None)


# ---- The run


def answers(count: int, outside_the_gap: int = 0) -> list:
    """Answers the grader labels as by hand, some of them with a score gap of 0.5."""
    return [
        a_case(marking("cp", 0, 0.75), marking("cp", 0, 0.25 if n < outside_the_gap else 0.75))
        for n in range(count)
    ]


def failed_gates(cases, rho: float = 0.95, kappa: float = 0.8) -> list[str]:
    return [gate.name for gate in gates(cases, measure(cases), rho, kappa) if not gate.passed]


def test_a_run_passes_with_nine_answers_in_ten_passing_each_metric() -> None:
    assert failed_gates(answers(10, outside_the_gap=1)) == []
    assert failed_gates(answers(10, outside_the_gap=2)) == ["Score gap"]


def test_a_run_needs_rho_and_kappa_over_their_thresholds() -> None:
    cases = answers(MIN_FOR_AGREEMENT)

    assert failed_gates(cases, rho=0.90, kappa=0.75) == []
    assert failed_gates(cases, rho=0.899) == [
        "Spearman ρ, the grader's score against the hand labels scored"
    ]
    assert failed_gates(cases, kappa=0.749) == ["Cohen's κ, key-point labels"]
    # Scores all the same give no correlation: that fails too.
    assert len(failed_gates(cases, rho=math.nan, kappa=math.nan)) == 2


def test_rho_and_kappa_gate_a_run_only_once_enough_answers_are_graded() -> None:
    """On a dozen answers, one answer can move ρ by a tenth."""
    fewer = answers(MIN_FOR_AGREEMENT - 1)
    graded_too_few = [
        *answers(MIN_FOR_AGREEMENT - 1),
        a_case(marking("cp", 0, 0.75), None, not_graded="grade it first"),
    ]

    assert failed_gates(fewer, rho=0.5, kappa=0.2) == []
    assert "Cohen's κ, key-point labels" not in [
        gate.name for gate in gates(fewer, measure(fewer), 0.5, 0.2)
    ]
    # Answers count once graded: an ungraded one does not bring ρ and κ in.
    assert failed_gates(graded_too_few, rho=0.5, kappa=0.2) == ["Every answer graded"]


def test_a_run_needs_every_injection_held_and_every_answer_graded() -> None:
    cases = answers(10)
    held = a_case(marking("mmm", 0, 0.0), marking("mmm", 0, 0.0), injection=True)
    talked_round = a_case(marking("mmm", 0, 0.0), marking("pmm", 0, 0.1), injection=True)
    ungraded = a_case(marking("cp", 0, 0.75), None, not_graded="grade it first")

    assert failed_gates([*cases, held]) == []
    assert failed_gates([*cases, held, talked_round]) == ["Injection held"]
    assert failed_gates([*cases, ungraded]) == ["Every answer graded"]


# ---- The calibration answers, replayed


def grade_line(key: str, labels: str, contradicted: int, score: float) -> str:
    return json.dumps(
        {
            "key": key,
            "prompt_version": PROMPT_VERSION,
            "question_id": 1,
            "model": "qwen/qwen3.8-27b",
            "labels": [NAMES[label] for label in labels],
            "contradicted": contradicted,
            # As saved when it was graded; the suite scores the labels again.
            "score": score,
        }
    )


def entry(question: int, text: str, labels: str, own: int, note: str = "") -> str:
    labels_toml = ", ".join(f'"{NAMES[label]}"' for label in labels)
    return (
        f'question = {question}\nnote = "{note}"\ntext = "{text}"\n'
        f"labels = [{labels_toml}]\ncontradicted = 0\nscore = {own}"
    )


def test_the_replay_scores_saved_grades_by_today_s_rules_and_never_asks_a_model(
    sessions, corpus, tmp_path, monkeypatch
) -> None:
    def no_model(*args, **kwargs):
        raise AssertionError("the replay asked for a model")

    monkeypatch.setattr(evaluate_grader, "groq_grader", no_model)
    folder = tmp_path / "calibration"
    folder.mkdir()

    async def scenario():
        question = await add_question(sessions, corpus.scaling)  # key point weights 2 and 1
        (folder / "answers.toml").write_text(
            answer_file(
                entry(question, "They are scaled down.", "cm", 7),
                entry(question, "Nobody knows.", "mm", 0),
            ),
            encoding="utf-8",
        )
        async with sessions() as session:
            first, second = evaluate_grader.load(
                (folder / "answers.toml").read_text(encoding="utf-8"),
                await evaluate_grader.calibration_questions(session),
            )
            # Saved with a score from rules since changed: 0.5 for labels worth 0.6667 today
            (folder / "grades.jsonl").write_text(
                grade_line(first.key, "cm", 0, 0.5) + "\n", encoding="utf-8"
            )
            return await calibration_suite(
                session, folder, live=False, settings=Settings(_env_file=None)
            )

    suite = asyncio.run(scenario())
    run = run_suite(suite)

    [first, second] = run.cases
    assert first.metadata["grader"]["score"] == 0.6667
    assert first.metadata["hand"]["score"] == 0.7
    assert second.metadata["grader"] is None
    assert "grade it first" in second.metadata["not_graded"]
    assert [gate.name for gate in run.gates if not gate.passed][0] == "Every answer graded"
    assert not run.passed


def test_the_calibration_suite_needs_an_answer_file(sessions, tmp_path) -> None:
    async def scenario():
        async with sessions() as session:
            await calibration_suite(session, tmp_path, live=False, settings=Settings())

    with pytest.raises(SuiteError, match="make calibrate ARGS=template"):
        asyncio.run(scenario())


def test_live_grading_grades_only_what_has_no_saved_grade(
    sessions, corpus, tmp_path, monkeypatch
) -> None:
    calls: list[int] = []
    output = {
        "key_points": [
            {"id": "k1", "status": "covered", "answer_quote": "scaled down"},
            {"id": "k2", "status": "missing", "answer_quote": ""},
        ],
        "claims": [],
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

    grader = FunctionModel(respond, profile=ModelProfile(supports_json_schema_output=True))
    monkeypatch.setattr(evaluate_grader, "groq_grader", lambda *args, **kwargs: grader)
    folder = tmp_path / "calibration"
    folder.mkdir()

    async def scenario():
        question = await add_question(sessions, corpus.scaling)
        (folder / "answers.toml").write_text(
            answer_file(
                entry(question, "They are scaled down.", "cm", 7),
                entry(question, "They are scaled down, as said.", "cm", 7),
            ),
            encoding="utf-8",
        )
        async with sessions() as session:
            first, _ = evaluate_grader.load(
                (folder / "answers.toml").read_text(encoding="utf-8"),
                await evaluate_grader.calibration_questions(session),
            )
            (folder / "grades.jsonl").write_text(
                grade_line(first.key, "cm", 0, 0.6667) + "\n", encoding="utf-8"
            )
            return await calibration_suite(
                session, folder, live=True, settings=Settings(_env_file=None), say=print
            )

    suite = asyncio.run(scenario())

    assert len(calls) == 1
    assert suite.graded_now == 1
    assert suite.notes[-1].endswith("tokens spent on 1 grade in this run.")
    assert len(read_results(folder / "grades.jsonl")) == 2
    assert run_suite(suite).passed


def test_the_live_suite_refuses_fake_models(monkeypatch, tmp_path, capsys) -> None:
    """A stand-in's grades measure nothing."""
    settings = Settings(_env_file=None, fake_models=True, data_dir=tmp_path)
    monkeypatch.setattr(evaluate_grader, "get_settings", lambda: settings)

    for chosen in ("calibration", "stand-ins"):
        args = argparse.Namespace(set=chosen, live=True)
        assert asyncio.run(evaluate_grader.run(args)) == 1
        assert "unset FAKE_MODELS" in capsys.readouterr().out


# ---- The stand-ins


@pytest.fixture(scope="module")
def stand_ins():
    """The stand-ins, their recorded outputs graded once for the module."""
    return asyncio.run(stand_in_suite(STAND_IN_FOLDER, model=None, say=print))


def test_the_stand_ins_cover_every_kind_of_answer() -> None:
    notes = [answer["note"] for answer in STAND_IN_ANSWERS]

    for kind in ("strong", "partial", "terse", "confidently wrong", "injection", "non-answer"):
        assert any(note.startswith(kind) for note in notes), kind


@pytest.mark.parametrize(
    "number",
    range(1, len(STAND_IN_ANSWERS) + 1),
    ids=[f"a{n} {answer['note'][:30]}" for n, answer in enumerate(STAND_IN_ANSWERS, 1)],
)
def test_every_stand_in_passes_every_metric_that_applies(stand_ins, number: int) -> None:
    case = list(grading_cases(stand_ins))[number - 1]

    assert_test(case, metrics_for(case), run_async=False)


def test_the_stand_in_suite_passes_through_the_grading_code(stand_ins) -> None:
    run = run_suite(stand_ins)

    assert run.passed
    assert [(gate.name, gate.measured) for gate in run.gates] == [
        ("Every answer graded", "12 of 12"),
        ("Key-point agreement", "12 of 12 (100%)"),
        ("Score gap", "12 of 12 (100%)"),
        ("Contradiction caught", "3 of 3 (100%)"),
        ("Injection held", "2 of 2"),
    ]
    # Twelve answers are too few for ρ and κ to gate the run: they are only reported.
    assert (round(run.agreement.rho, 3), round(run.agreement.kappa, 3)) == (0.925, 0.906)
    # Every grade came from a recorded output, through the same code as a live grade.
    assert {result["model"] for result in stand_ins.results.values()} == {"recorded"}


def partial_worth_full(monkeypatch) -> None:
    monkeypatch.setitem(scoring.CREDIT, "partial", 1.0)


def contradictions_free(monkeypatch) -> None:
    monkeypatch.setattr(scoring, "CONTRADICTION_PENALTY", 0.0)


@pytest.mark.parametrize(
    ("break_scoring", "failed_answers", "failed_gates"),
    [
        (
            partial_worth_full,
            {("a2 q1", "Score gap"), ("a9 q3", "Score gap"), ("a10 q3", "Injection held")},
            ["Score gap", "Injection held"],
        ),
        (
            contradictions_free,
            {("a3 q1", "Score gap"), ("a12 q3", "Score gap")},
            ["Score gap"],
        ),
    ],
    ids=["a partial label worth a full one", "contradictions costing nothing"],
)
def test_a_broken_scoring_rule_fails_the_stand_in_suite(
    monkeypatch, break_scoring, failed_answers, failed_gates
) -> None:
    break_scoring(monkeypatch)

    run = run_suite(asyncio.run(stand_in_suite(STAND_IN_FOLDER, model=None, say=print)))

    assert not run.passed
    assert {(outcome.answer, outcome.metric) for outcome in run.outcomes if not outcome.passed} == (
        failed_answers
    )
    assert [gate.name for gate in run.gates if not gate.passed] == failed_gates


def test_the_live_stand_ins_stop_once_the_day_is_spent() -> None:
    calls: list[int] = []
    first = STAND_IN_ANSWERS[0]["grader_output"]

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(1)
        if len(calls) > 1:
            raise QuotaExhausted("groq-grading", "groq-grading says it is out of quota for today")
        return ModelResponse(parts=[TextPart(json.dumps(first))])

    model = FunctionModel(respond, profile=ModelProfile(supports_json_schema_output=True))

    suite = asyncio.run(stand_in_suite(STAND_IN_FOLDER, model=model, say=print))

    # One graded, and one request spent finding out the day is over.
    assert (len(calls), len(suite.results), suite.graded_now) == (2, 1, 1)
    assert suite.notes[-1].endswith("tokens spent on 1 grade in this run.")
    assert set(suite.not_graded.values()) == {"the grader is out of quota for today"}
    assert len(suite.not_graded) == len(STAND_IN_ANSWERS) - 1
    assert not run_suite(suite).passed


def test_the_stand_ins_run_from_the_command_line(capsys) -> None:
    args = argparse.Namespace(set="stand-ins", live=False)

    assert asyncio.run(evaluate_grader.run(args)) == 0
    report = capsys.readouterr().out
    assert (
        "12 answers to 3 questions, the stand-ins in backend/tests/fixtures/grader_suite" in report
    )
    assert "**Passed.**" in report

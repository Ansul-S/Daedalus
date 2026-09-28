"""Check the grader against answers graded by hand: the grader's regression suite.

Run from the repo root:
    make eval-grader                        the calibration answers, from their saved grades
    make eval-grader LIVE=1                 the same, grading first what has no saved grade
    make eval-grader SET=stand-ins          the stand-ins in backend/tests/fixtures/grader_suite
    make eval-grader SET=stand-ins LIVE=1   the stand-ins, graded by the real grader

The calibration answers are the hand grades in data/calibration (see scripts/calibrate.py).
They are measured from the grades saved under the current prompt version, scored by today's
rules. An answer with no saved grade fails with "grade it first", and no model is asked. With
LIVE, the grader alone grades what is missing first, as `make calibrate` does: no fallback,
and a stop once the provider says the day is spent. The new grades are saved with the others.

The stand-ins are answers written for the repository, each with its hand grade and the output
the grader would send back for it. Replayed, that output goes through the same code as a live
grade, so the suite checks everything between the model and the score anywhere, CI included,
without anyone's practice data. With LIVE, the real grader grades them afresh (about 20K
tokens, which the report counts) and nothing is saved; with no database to ask, the pacer does
not know what was spent earlier in the day, and relies on the provider to say when it is.

The metrics, and what a run has to meet, are in app/evaluation/grader_metrics.py. The report
is printed, and saved to data/reports/grader-<date>.md for the calibration answers. The script
exits with 1 when the run does not pass.
"""

import argparse
import asyncio
import json
import logging
import math
import sys
import tomllib
import warnings
from collections import Counter
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from pydantic_ai.exceptions import AgentRunError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.models import Question
from app.db.session import SessionFactory, engine
from app.evaluation.grader_metrics import (
    MIN_FOR_AGREEMENT,
    Case,
    Gate,
    Marking,
    Outcome,
    agreement_gated,
    gates,
    grading_case,
    is_injection,
    measure,
    short,
)
from app.grading.grader import PROMPT_VERSION, grade_answer, spent_today, why_failed
from app.grading.scoring import score
from app.llm.models import groq_grader
from app.llm.tracing import traced, tracing
from app.questions.batch import out_of_budget
from app.questions.generation import Source
from scripts.calibrate import (
    Agreement,
    AnswerFileError,
    HandGrade,
    agreement,
    calibration_questions,
    grade_all,
    load,
    read_results,
    say,
)

CALIBRATION, STAND_INS = "calibration", "stand-ins"
REPO = Path(__file__).resolve().parents[2]
STAND_IN_FOLDER = REPO / "backend" / "tests" / "fixtures" / "grader_suite"
# A stand-in for the grader that sends back the output written down for an answer
RECORDED = "recorded"


def shown(path: Path) -> str:
    """A path as the report gives it: from the repository's root when it is inside it."""
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


class SuiteError(Exception):
    """The suite cannot run; the message says what to do."""


@dataclass
class Suite:
    """Hand-graded answers and the grader's grades of them, scored by today's rules."""

    name: str
    where: str
    hand: list[HandGrade]
    weights: dict[int, list[int]]
    # The grader's grades by answer and prompt version, as `scripts.calibrate` keeps them
    results: dict[tuple[str, str], dict[str, Any]]
    # Why an answer has no grade, by its key
    not_graded: dict[str, str] = field(default_factory=dict)
    graded_now: int = 0
    notes: list[str] = field(default_factory=list)


def scored(result: dict[str, Any], weights: list[int]) -> dict[str, Any]:
    """A grade with the score today's rules give its labels."""
    return result | {"score": score(weights, result["labels"], result["contradicted"]).score}


def weights_of(questions: dict[int, Question]) -> dict[int, list[int]]:
    return {qid: [int(p["weight"]) for p in q.key_points] for qid, q in questions.items()}


def tokens(usage: dict[str, int] | None) -> int:
    usage = usage or {}
    return usage.get("input_tokens", 0) + usage.get("output_tokens", 0)


def spent(tokens_used: int, grades: int) -> str:
    return (
        f"{tokens_used:,} tokens spent on {grades} grade{'' if grades == 1 else 's'} in this run."
    )


# ---- The calibration answers


async def calibration_suite(
    session: AsyncSession, folder: Path, *, live: bool, settings: Settings, say=say
) -> Suite:
    answers_path, results_path = folder / "answers.toml", folder / "grades.jsonl"
    if not answers_path.exists():
        raise SuiteError(
            f"No {answers_path}: write hand grades with `make calibrate ARGS=template`, "
            "or run the stand-ins with `make eval-grader SET=stand-ins`."
        )
    questions = await calibration_questions(session)
    hand = load(answers_path.read_text(encoding="utf-8"), questions)
    notes, before = [], set(read_results(results_path))
    if live:
        model = groq_grader(settings, await spent_today(session))
        if model is None:
            raise SuiteError("The suite measures the Groq grader: set GROQ_API_KEY.")
        left = await grade_all(session, model, hand, questions, results_path, say=say)
        if left:
            notes.append(f"{left} answers could not be graded in this run.")
    weights = weights_of(questions)
    saved = read_results(results_path)
    results, not_graded = {}, {}
    for grade in hand:
        key = (grade.key, PROMPT_VERSION)
        if key in saved:
            results[key] = scored(saved[key], weights[grade.question_id])
        else:
            not_graded[grade.key] = (
                f"no grade saved under {PROMPT_VERSION}: grade it first with "
                "`make eval-grader LIVE=1`"
            )
    graded_now = set(saved) - before
    if live:
        notes.append(
            spent(sum(tokens(saved[key].get("usage")) for key in graded_now), len(graded_now))
        )
    return Suite(
        name=CALIBRATION,
        where=f"the hand grades in {shown(folder)}",
        hand=hand,
        weights=weights,
        results=results,
        not_graded=not_graded,
        graded_now=len(graded_now & set(results)),
        notes=notes,
    )


# ---- The stand-ins


def stand_in_questions(folder: Path) -> tuple[dict[int, Question], dict[int, list[Source]]]:
    """The stand-in questions, and each one's passages in the order they are shown."""
    rows = tomllib.loads((folder / "questions.toml").read_text(encoding="utf-8"))["question"]
    questions = {
        row["id"]: Question(
            id=row["id"], text=row["text"], key_points=row["key_points"], status="accepted"
        )
        for row in rows
    }
    sources = {
        row["id"]: [
            Source(passage["chunk_id"], passage["citation"], passage["text"].strip())
            for passage in row["passage"]
        ]
        for row in rows
    }
    return questions, sources


def recorded(output: dict[str, Any]) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(json.dumps(output))], model_name=RECORDED)

    return FunctionModel(
        respond, model_name=RECORDED, profile=ModelProfile(supports_json_schema_output=True)
    )


async def stand_in_suite(folder: Path, *, model: Model | None, say=say) -> Suite:
    """The stand-ins, graded by `model`, or, without one, by their recorded outputs."""
    questions, sources = stand_in_questions(folder)
    text = (folder / "answers.toml").read_text(encoding="utf-8")
    hand = load(text, questions)
    outputs = [entry.get("grader_output") for entry in tomllib.loads(text)["answer"]]
    results: dict[tuple[str, str], dict[str, Any]] = {}
    not_graded: dict[str, str] = {}
    used = 0
    for count, (grade, output) in enumerate(zip(hand, outputs, strict=True), 1):
        grader = model or (recorded(output) if output is not None else None)
        if grader is None:
            not_graded[grade.key] = "no grader_output written down for it"
            continue
        question = questions[grade.question_id]
        try:
            with traced(
                "grade", version=PROMPT_VERSION, question=question.id, stand_in=grade.number
            ):
                graded = await grade_answer(
                    grader, question.text, question.key_points, sources[question.id], grade.text
                )
        except (AgentRunError, ExceptionGroup) as exc:
            if out_of_budget(exc):
                for rest in hand[count - 1 :]:
                    not_graded[rest.key] = "the grader is out of quota for today"
                say(f"Out of quota for today ({exc}); the rest are not graded.")
                break
            not_graded[grade.key] = f"the grade failed: {why_failed(exc)}"
            continue
        labels = [point["status"] for point in graded.key_points]
        results[(grade.key, PROMPT_VERSION)] = {
            "labels": labels,
            "contradicted": graded.result.contradicted,
            "score": graded.result.score,
            "model": graded.model,
        }
        if model is not None:
            used += tokens(graded.usage)
            say(
                f"{count}/{len(hand)} a{grade.number} q{grade.question_id}: {short(labels)} -> "
                f"{graded.result.score:.2f} (by hand {short(grade.labels)}), "
                f"{graded.seconds:.1f} s, {tokens(graded.usage):,} tokens, {graded.model}"
            )
    return Suite(
        name=STAND_INS,
        where=f"the stand-ins in {shown(folder)}",
        hand=hand,
        weights=weights_of(questions),
        results=results,
        not_graded=not_graded,
        graded_now=len(results) if model is not None else 0,
        notes=[spent(used, len(results))] if model is not None else [],
    )


# ---- The run


def answer_name(grade: HandGrade) -> str:
    return f"a{grade.number} q{grade.question_id}"


def grading_cases(suite: Suite) -> Iterator[Case]:
    for grade in suite.hand:
        weights = suite.weights[grade.question_id]
        if grade.score is not None:
            hand_score = grade.score / 10
        else:
            hand_score = score(weights, grade.labels, grade.contradicted).score
        found = suite.results.get((grade.key, PROMPT_VERSION))
        grader = (
            Marking(tuple(found["labels"]), found["contradicted"], found["score"])
            if found
            else None
        )
        yield grading_case(
            answer_name(grade),
            grade.text,
            Marking(grade.labels, grade.contradicted, hand_score),
            grader,
            own_score=grade.score is not None,
            injection=is_injection(grade.note),
            not_graded=suite.not_graded.get(grade.key, ""),
        )


@dataclass
class Run:
    suite: Suite
    cases: list[Case]
    outcomes: list[Outcome]
    agreement: Agreement | None
    gates: list[Gate]

    @property
    def passed(self) -> bool:
        return all(gate.passed for gate in self.gates)


def run_suite(suite: Suite) -> Run:
    cases = list(grading_cases(suite))
    outcomes = measure(cases)
    with warnings.catch_warnings():
        # With only a grade or two, κ is undefined: its gate says so, without sklearn's warnings.
        warnings.simplefilter("ignore", UserWarning)
        found = agreement(suite.hand, suite.results, suite.weights)
    rho, kappa = (found.rho, found.kappa) if found else (math.nan, math.nan)
    return Run(suite, cases, outcomes, found, gates(cases, outcomes, rho, kappa))


# ---- The report


def number(value: float | None) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:.3f}"


def cell(text: str, width: int | None = None) -> str:
    """Text for a table cell: no line breaks or pipes, and cut at a word when too long."""
    text = " ".join(text.split()).replace("|", "\\|")
    if width is None or len(text) <= width:
        return text
    return text[:width].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def render(run: Run, day: date) -> str:
    suite, found = run.suite, run.agreement
    questions = len({grade.question_id for grade in suite.hand})
    models = Counter(result["model"] for result in suite.results.values())
    by = ", ".join(f"{model} ({count})" for model, count in models.items()) or "no model"
    if suite.name == STAND_INS and not suite.graded_now:
        how = "the grader's output written down for each, replayed through the grading code"
    else:
        how = f"graded under {PROMPT_VERSION} by {by}"
        if suite.name == CALIBRATION:
            how += (
                f", {suite.graded_now} of them in this run and the rest from their saved grades"
                if suite.graded_now
                else ", replayed from their saved grades"
            )
    lines = [
        f"# Grader suite, {day.isoformat()}",
        "",
        f"{len(suite.hand)} answers to {questions} questions, {suite.where}: {how}. Every score "
        "comes from the labels, by today's rules.",
        "",
        "## Gates",
        "",
        "| Gate | Measured | Needs | |",
        "|---|---|---|---|",
        *(
            f"| {gate.name} | {gate.measured} | {gate.needs} | "
            f"{'pass' if gate.passed else '**fail**'} |"
            for gate in run.gates
        ),
        "",
        "**Passed.**"
        if run.passed
        else "**Failed:** " + "; ".join(gate.name for gate in run.gates if not gate.passed) + ".",
    ]
    notes = {answer_name(grade): grade.note for grade in suite.hand}
    failed = [outcome for outcome in run.outcomes if not outcome.passed]
    lines += ["", "## Answers that failed a metric", ""]
    if failed:
        lines += ["| Answer | Note | Metric | Why |", "|---|---|---|---|"]
        lines += [
            f"| {outcome.answer} | {cell(notes[outcome.answer], 50)} | {outcome.metric} "
            f"| {cell(outcome.reason)} |"
            for outcome in failed
        ]
    else:
        lines.append("None.")
    if found is not None or suite.notes:
        lines += ["", "## Beside the gates", ""]
    if found is not None:
        swaps = ", ".join(
            f"{hand}→{graded} {count}"
            for (hand, graded), count in sorted(found.confusion.items())
            if hand != graded
        )
        by_whom = ", ".join(f"{who} {count}" for who, count in sorted(found.contradiction.items()))
        if not agreement_gated(run.cases):
            lines.append(
                f"- Spearman ρ {number(found.rho)} and Cohen's κ {number(found.kappa)}, not "
                f"gated: they gate a run from {MIN_FOR_AGREEMENT} graded answers, and "
                f"{found.answers} are graded."
            )
        lines += [
            f"- Key-point labels {found.label_agreement:.0%} the same (linear-weighted κ "
            f"{number(found.kappa_linear)}); where they differ, the hand's → the grader's: "
            f"{swaps or 'nowhere'}.",
            f"- Answers with a contradicted claim, found by: {by_whom}.",
            f"- Mean score difference from the hand labels scored: {found.mean_difference:.3f}.",
        ]
        if found.own_count:
            lines.append(
                f"- Against the hand's own scores ({found.own_count} answers): Spearman ρ "
                f"{number(found.rho_own)} for the grader, {number(found.rho_formula_own)} for "
                "the hand labels scored."
            )
    lines += [f"- {note}" for note in suite.notes]
    return "\n".join(lines) + "\n"


# ---- Running it


async def main(args: argparse.Namespace) -> int:
    try:
        return await run(args)
    finally:
        await engine.dispose()


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    if args.live and settings.fake_models:
        print("The suite measures the real grader: unset FAKE_MODELS to grade.")
        return 1
    try:
        if args.set == STAND_INS:
            model = None
            if args.live:
                model = groq_grader(settings)
                if model is None:
                    print("The suite measures the Groq grader: set GROQ_API_KEY.")
                    return 1
            suite = await stand_in_suite(STAND_IN_FOLDER, model=model)
        else:
            async with SessionFactory() as session:
                suite = await calibration_suite(
                    session, settings.data_dir / "calibration", live=args.live, settings=settings
                )
    except (SuiteError, AnswerFileError) as exc:
        print(exc)
        return 1
    day = date.today()
    result = run_suite(suite)
    report = render(result, day)
    print(report)
    if suite.name == CALIBRATION:
        path = settings.data_dir / "reports" / f"grader-{day.isoformat()}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
        print(f"Saved to {path}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the grader against hand grades.")
    parser.add_argument("--set", choices=[CALIBRATION, STAND_INS], default=CALIBRATION)
    parser.add_argument("--live", action="store_true", help="grade with the real grader first")
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # Waits and retries, so a slow stretch explains itself
    logging.getLogger("app.llm.pacing").setLevel(logging.INFO)
    arguments = parser.parse_args()
    # A replay calls no model, and its stand-in grader is not worth a trace
    with tracing(get_settings(), "eval-grader") if arguments.live else nullcontext():
        sys.exit(asyncio.run(main(arguments)))

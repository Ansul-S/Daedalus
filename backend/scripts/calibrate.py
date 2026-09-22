"""Measure how far the grader agrees with grades given by hand.

Run from the repo root:
    make calibrate ARGS=template   write data/calibration/answers.toml to fill in
    make calibrate                 grade every answer in it not graded yet, then report
    make calibrate ARGS=report     report on the grades already made, without grading

Each answer in the file carries the grader's labels as they should be: one per key point,
covered, partial or missing, in the key points' order; how many of its claims contradict
the sources; and, optionally, an overall score out of 10. The grader's score is compared with
the score the same formula gives those hand labels (Spearman rank correlation; scores are
trusted from 0.7), its key-point labels with the hand labels (Cohen's kappa), and its
contradictions with the hand count. An answer keeps counting after its question is retired
from practice: it was graded against the question as it stood, whose passages are still there.

The answers are graded by the grader alone, with no fallback: a grade from another model
would measure that model instead. When the provider says the day is spent, grading stops and
the rest waits for the next run. Grades are kept in data/calibration/grades.jsonl, one per
answer and prompt version, so running again grades only what is new. The file is personal
practice data and stays out of the repository with the rest of data/.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import math
import sys
import tomllib
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic_ai.exceptions import AgentRunError
from pydantic_ai.models import Model
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.models import Question
from app.db.session import SessionFactory, engine
from app.grading.grader import PROMPT_VERSION, grade_answer, question_sources, spent_today
from app.grading.scoring import score
from app.llm.models import groq_grader
from app.questions.batch import out_of_budget

STATUSES = ("covered", "partial", "missing")
# Scores are trusted once they rank answers the way the hand grades do this well.
TRUSTED_RHO = 0.7

HEADER = """\
# Hand grades for calibrating the grader. Add one [[answer]] table per answer, under the
# question it answers or anywhere else; the order does not matter.
#
#   question      the question's id, from the comment above it
#   text          the answer, as a candidate would give it
#   labels        one per key point, in the order below: "covered", "partial" or "missing"
#   contradicted  how many claims in the answer contradict the sources
#   score         optional: your own overall score out of 10
#   note          optional: anything, e.g. "strong", "confidently wrong", "injection"
#
# [[answer]]
# question = 31
# note = "partial"
# text = \"\"\"
# Because in a sequence the meaning depends on what came before.
# \"\"\"
# labels = ["covered", "missing", "partial"]
# contradicted = 0
# score = 5
#
# A question's reference answer and sources: GET /questions/<id> on the API.
"""


class AnswerFileError(ValueError):
    """The answer file has problems; every one found is listed."""


@dataclass(frozen=True)
class HandGrade:
    number: int  # its place in the file, counting from 1
    question_id: int
    text: str
    labels: tuple[str, ...]
    contradicted: int
    score: float | None
    note: str

    @property
    def key(self) -> str:
        """Identifies the answer as graded: the same question and the same words."""
        digest = hashlib.sha256(f"{self.question_id}\n{self.text.strip()}".encode()).hexdigest()
        return digest[:16]


def say(message: str) -> None:
    # Flushed, so progress shows as it happens even when the output goes to a file
    print(message, flush=True)


def calibration_dir() -> Path:
    return get_settings().data_dir / "calibration"


async def accepted_questions(session: AsyncSession) -> dict[int, Question]:
    rows = await session.scalars(
        select(Question).where(Question.status == "accepted").order_by(Question.id)
    )
    return {question.id: question for question in rows}


async def calibration_questions(session: AsyncSession) -> dict[int, Question]:
    """The questions answers are graded against: those in the library, and those retired from
    it since. No answer is written for a rejected one."""
    rows = await session.scalars(
        select(Question).where(Question.status.in_(("accepted", "retired"))).order_by(Question.id)
    )
    return {question.id: question for question in rows}


def template(questions: dict[int, Question]) -> str:
    blocks = [HEADER]
    for question in questions.values():
        lines = [f"# q{question.id} [{question.style}] {question.text}"]
        lines += [
            f"#   k{number} (weight {point['weight']}) {point['text']}"
            for number, point in enumerate(question.key_points, 1)
        ]
        blocks.append("\n".join(lines) + "\n")
    return "\n".join(blocks)


def write_template(path: Path, questions: dict[int, Question]) -> bool:
    """Write the template unless the file is already there. Returns whether it wrote."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template(questions), encoding="utf-8")
    return True


def load(text: str, questions: dict[int, Question]) -> list[HandGrade]:
    """The hand grades in an answer file, or an error listing everything wrong with it."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise AnswerFileError(f"not valid TOML: {exc}") from exc
    problems: list[str] = []
    grades: list[HandGrade] = []
    for number, entry in enumerate(data.get("answer", []), 1):
        before = len(problems)
        where = f"answer {number}"
        question_id = entry.get("question")
        question = questions.get(question_id) if isinstance(question_id, int) else None
        if question is None:
            problems.append(
                f"{where}: question {question_id!r} is not an accepted or retired question"
            )
            continue
        where = f"answer {number} (q{question_id})"
        answer = entry.get("text")
        if not isinstance(answer, str) or not answer.strip():
            problems.append(f"{where}: text is empty")
        labels = entry.get("labels")
        if not isinstance(labels, list) or len(labels) != len(question.key_points):
            problems.append(f"{where}: needs {len(question.key_points)} labels, one per key point")
        elif bad := [label for label in labels if label not in STATUSES]:
            problems.append(f"{where}: labels must be covered, partial or missing, not {bad}")
        contradicted = entry.get("contradicted")
        if not isinstance(contradicted, int) or isinstance(contradicted, bool) or contradicted < 0:
            problems.append(f"{where}: contradicted must be a whole number, 0 or more")
        own = entry.get("score")
        if own is not None and (
            not isinstance(own, int | float) or isinstance(own, bool) or not 0 <= own <= 10
        ):
            problems.append(f"{where}: score must be between 0 and 10")
        if len(problems) == before:
            grades.append(
                HandGrade(
                    number=number,
                    question_id=question_id,
                    text=answer,
                    labels=tuple(labels),
                    contradicted=contradicted,
                    score=float(own) if own is not None else None,
                    note=str(entry.get("note", "")),
                )
            )
    if problems:
        raise AnswerFileError("\n".join(problems))
    return grades


def read_results(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Grades made so far, by answer and prompt version."""
    if not path.exists():
        return {}
    results = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            result = json.loads(line)
            results[(result["key"], result["prompt_version"])] = result
    return results


async def grade_all(
    session: AsyncSession,
    model: Model,
    hand: list[HandGrade],
    questions: dict[int, Question],
    results_path: Path,
    say=say,
) -> int:
    """Grade every answer that has no grade under the current prompt version yet. A grade
    that failed is not written down, so the next run tries again, and once the provider is out
    of quota for the day nothing more is tried. Returns how many are left ungraded."""
    done = read_results(results_path)
    todo = [grade for grade in hand if (grade.key, PROMPT_VERSION) not in done]
    say(f"{len(hand)} answers, {len(hand) - len(todo)} graded already, {len(todo)} to grade.")
    failed = 0
    for count, grade in enumerate(todo, 1):
        question = questions[grade.question_id]
        sources = await question_sources(session, question.id)
        try:
            graded = await grade_answer(
                model, question.text, question.key_points, sources, grade.text
            )
        except (AgentRunError, ExceptionGroup) as exc:
            if out_of_budget(exc):
                left = len(todo) - count + 1
                say(f"Out of quota for today ({exc}); {left} left to grade on the next run.")
                return failed + left
            failed += 1
            say(f"{count}/{len(todo)} q{grade.question_id} answer {grade.number}: failed ({exc})")
            continue
        labels = [point["status"] for point in graded.key_points]
        result = {
            "key": grade.key,
            "prompt_version": PROMPT_VERSION,
            "question_id": grade.question_id,
            "model": graded.model,
            "labels": labels,
            "contradicted": graded.result.contradicted,
            "coverage": graded.result.coverage,
            "score": graded.result.score,
            "clarity": graded.clarity,
            "claims": graded.claims,
            "seconds": graded.seconds,
            "usage": graded.usage,
            "graded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        with results_path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(result) + "\n")
        say(
            f"{count}/{len(todo)} q{grade.question_id} answer {grade.number}: "
            f"{short(labels)} -> {graded.result.score:.2f} "
            f"(by hand {short(grade.labels)}), {graded.seconds:.1f} s, {graded.model}"
        )
    return failed


def short(labels) -> str:
    return "".join(label[0] for label in labels)


@dataclass
class Agreement:
    answers: int
    # Grader's score against the formula applied to the hand labels
    rho: float
    # Grader's score, and the formula on the hand labels, against the hand's own 0-10
    rho_own: float | None
    rho_formula_own: float | None
    own_count: int
    kappa: float
    kappa_linear: float
    label_agreement: float
    confusion: dict[tuple[str, str], int]
    # Answers the hand, the grader, or both, found something contradicted in
    contradiction: dict[str, int]
    mean_difference: float
    worst: list[tuple[HandGrade, dict[str, Any], float]]
    models: Counter

    @property
    def trusted(self) -> bool:
        return not math.isnan(self.rho) and self.rho >= TRUSTED_RHO


def agreement(
    hand: list[HandGrade], results: dict[tuple[str, str], dict[str, Any]], weights: dict
) -> Agreement | None:
    """How the grader's grades compare with the hand grades. Needs scipy and scikit-learn,
    from the eval dependency group."""
    from scipy.stats import spearmanr
    from sklearn.metrics import cohen_kappa_score

    pairs = [
        (grade, results[(grade.key, PROMPT_VERSION)])
        for grade in hand
        if (grade.key, PROMPT_VERSION) in results
    ]
    if not pairs:
        return None
    expected = [
        score(weights[grade.question_id], grade.labels, grade.contradicted).score
        for grade, _ in pairs
    ]
    graded = [result["score"] for _, result in pairs]

    def rho(first: list[float], second: list[float]) -> float:
        if len(first) < 3 or len(set(first)) < 2 or len(set(second)) < 2:
            return math.nan
        return float(spearmanr(first, second).statistic)

    own = [
        (grade.score, result["score"], e)
        for (grade, result), e in zip(pairs, expected, strict=True)
        if grade.score is not None
    ]
    hand_labels = [label for grade, _ in pairs for label in grade.labels]
    grader_labels = [label for _, result in pairs for label in result["labels"]]
    ordinal = {"missing": 0, "partial": 1, "covered": 2}
    contradiction = Counter(
        {
            (True, True): "both",
            (True, False): "hand only",
            (False, True): "grader only",
            (False, False): "neither",
        }[(grade.contradicted > 0, result["contradicted"] > 0)]
        for grade, result in pairs
    )
    gaps = [
        (grade, result, result["score"] - e)
        for (grade, result), e in zip(pairs, expected, strict=True)
    ]
    return Agreement(
        answers=len(pairs),
        rho=rho(expected, graded),
        rho_own=rho([o for o, _, _ in own], [g for _, g, _ in own]) if own else None,
        rho_formula_own=rho([o for o, _, _ in own], [e for _, _, e in own]) if own else None,
        own_count=len(own),
        kappa=float(cohen_kappa_score(hand_labels, grader_labels)),
        kappa_linear=float(
            cohen_kappa_score(
                [ordinal[label] for label in hand_labels],
                [ordinal[label] for label in grader_labels],
                weights="linear",
            )
        ),
        label_agreement=sum(a == b for a, b in zip(hand_labels, grader_labels, strict=True))
        / len(hand_labels),
        confusion=Counter(zip(hand_labels, grader_labels, strict=True)),
        contradiction=dict(contradiction),
        mean_difference=sum(abs(gap) for _, _, gap in gaps) / len(gaps),
        worst=sorted(gaps, key=lambda item: -abs(item[2]))[:5],
        models=Counter(result["model"] for _, result in pairs),
    )


def report(found: Agreement | None, total: int, say=say) -> None:
    if found is None:
        say("Nothing graded yet: run `make calibrate` to grade the answers.")
        return

    def number(value: float | None) -> str:
        return "n/a" if value is None or math.isnan(value) else f"{value:.2f}"

    models = ", ".join(f"{model} {count}" for model, count in found.models.items())
    say(f"\n{found.answers} of {total} answers graded under {PROMPT_VERSION} ({models})")
    verdict = "trusted" if found.trusted else f"not yet trusted (needs {TRUSTED_RHO})"
    say(f"Spearman rho, grader's score vs your labels scored: {number(found.rho)} -- {verdict}")
    if found.own_count:
        say(
            f"Against your own scores ({found.own_count} answers): grader {number(found.rho_own)}, "
            f"the formula on your labels {number(found.rho_formula_own)}"
        )
    say(
        f"Key-point labels: {found.label_agreement:.0%} the same, Cohen's kappa "
        f"{found.kappa:.2f} (linear-weighted {found.kappa_linear:.2f})"
    )
    swaps = {
        f"{hand}->{graded}": n
        for (hand, graded), n in sorted(found.confusion.items())
        if hand != graded
    }
    say(f"  where they differ (yours->grader's): {swaps or 'nowhere'}")
    say(f"Contradictions found by: {found.contradiction}")
    say(f"Mean score difference: {found.mean_difference:.2f}")
    say("Largest differences:")
    for grade, result, gap in found.worst:
        say(
            f"  answer {grade.number} q{grade.question_id} {grade.note or ''}: yours "
            f"{short(grade.labels)} c{grade.contradicted}, grader's {short(result['labels'])} "
            f"c{result['contradicted']}, score {gap:+.2f}"
        )


async def main(args: argparse.Namespace) -> int:
    try:
        return await run(args)
    finally:
        await engine.dispose()


async def run(args: argparse.Namespace) -> int:
    folder = calibration_dir()
    answers_path, results_path = folder / "answers.toml", folder / "grades.jsonl"
    async with SessionFactory() as session:
        if args.command == "template":
            questions = await accepted_questions(session)
            if write_template(answers_path, questions):
                print(f"Wrote {answers_path} with {len(questions)} questions; add your answers.")
                return 0
            print(f"{answers_path} is there already; it is not overwritten.")
            return 1
        questions = await calibration_questions(session)
        if not answers_path.exists():
            print(f"No {answers_path}: start with `make calibrate ARGS=template`.")
            return 1
        try:
            hand = load(answers_path.read_text(encoding="utf-8"), questions)
        except AnswerFileError as exc:
            print(f"{answers_path}:\n{exc}")
            return 1
        failed = 0
        if args.command == "grade":
            model = groq_grader(get_settings(), await spent_today(session))
            if model is None:
                print("Grading is measured on the Groq grader: set GROQ_API_KEY.")
                return 1
            failed = await grade_all(session, model, hand, questions, results_path)
    weights = {qid: [int(p["weight"]) for p in q.key_points] for qid, q in questions.items()}
    report(agreement(hand, read_results(results_path), weights), len(hand))
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure the grader against hand grades.")
    parser.add_argument(
        "command", nargs="?", default="grade", choices=["template", "grade", "report"]
    )
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # Waits and retries, so a slow stretch explains itself
    logging.getLogger("app.llm.pacing").setLevel(logging.INFO)
    sys.exit(asyncio.run(main(parser.parse_args())))

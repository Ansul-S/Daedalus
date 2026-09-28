"""Retrieval evaluation without labelling every question by hand.

A generated question was written from one or two passages, and those passages are its answers:
search has found them when they rank near the top. Three sets of questions are measured apart,
never pooled:

- **asked**: the questions practice serves (accepted, and retired ones);
- **passage**: those plus the questions turned down for reasons that leave them about their
  passage (duplicate, trivia, a quote not found). A question the checker could not answer from
  its own passage may reach beyond it, so it is left out;
- **hand**: the 20 questions written by hand for Phase 1, with passages labelled by hand. They
  borrow no words from a passage, but they did guide the chunking and ranking choices.

A question's own passages undercount: another passage often answers it as well. Passages
confirmed by hand to answer a question (the "also answers" file) count in the lenient score,
beside the strict one.
"""

import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import mean

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Question, QuestionSource
from app.retrieval.search import CANDIDATES, KEYWORD_CANDIDATES, KEYWORD_WEIGHT, fuse_rankings

ASKED, PASSAGE, HAND = "asked", "passage", "hand"
SETS = (ASKED, PASSAGE, HAND)
# Rank cut-off of every score: the top five is what a person reads of a search.
TOP = 5
VERDICTS = ("yes", "partly", "no")


class LabelFileError(ValueError):
    pass


@dataclass(frozen=True)
class Query:
    id: str  # "q24" for a generated question, the file's id ("C2.3") for a hand-written one
    text: str
    gold: frozenset[int]
    sets: frozenset[str]


@dataclass(frozen=True)
class Scores:
    questions: int
    hit1: int
    hit5: int
    recall5: float
    mrr5: float


@dataclass(frozen=True)
class Fusion:
    """How the two rankings are merged: how many full-text candidates are kept, and their
    weight against the vector ranking's 1."""

    keyword_candidates: int = KEYWORD_CANDIDATES
    keyword_weight: float = KEYWORD_WEIGHT

    @property
    def label(self) -> str:
        return f"full-text {self.keyword_candidates} x {self.keyword_weight:g}"


CURRENT = Fusion()
# Fixed before any variant was measured (28 Sep 2026). Full share (50 x 1) was the setting then.
FUSION_GRID = tuple(
    Fusion(candidates, weight) for candidates in (CANDIDATES, 20, 10) for weight in (1.0, 0.5)
)


async def generated_queries(session: AsyncSession) -> tuple[list[Query], int]:
    """The asked and passage sets, and how many questions were left out as possibly reaching
    beyond their passage."""
    questions = (await session.scalars(select(Question).order_by(Question.id))).all()
    sources = (
        await session.execute(select(QuestionSource.question_id, QuestionSource.chunk_id))
    ).all()
    gold: dict[int, set[int]] = {}
    for question_id, chunk_id in sources:
        gold.setdefault(question_id, set()).add(chunk_id)
    queries, left_out = [], 0
    for question in questions:
        if question.id not in gold:
            continue
        if question.status in ("accepted", "retired"):
            sets = {ASKED, PASSAGE}
        elif "answerable" in question.validation.get("failed", []):
            left_out += 1
            continue
        else:
            sets = {PASSAGE}
        queries.append(
            Query(f"q{question.id}", question.text, frozenset(gold[question.id]), frozenset(sets))
        )
    return queries, left_out


def hand_queries(text: str) -> list[Query]:
    try:
        rows = tomllib.loads(text).get("question", [])
    except tomllib.TOMLDecodeError as exc:
        raise LabelFileError(str(exc)) from exc
    queries = []
    for number, row in enumerate(rows, start=1):
        try:
            queries.append(
                Query(
                    str(row["id"]),
                    str(row["text"]),
                    frozenset(map(int, row["gold"])),
                    frozenset({HAND}),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LabelFileError(
                f"question {number}: needs id, text and a list of gold chunk ids"
            ) from exc
    ids = [query.id for query in queries]
    if len(set(ids)) != len(ids):
        raise LabelFileError("question ids must be unique")
    return queries


def also_answers(text: str) -> dict[str, frozenset[int]]:
    """The passages confirmed to answer a question beyond its own, by question id."""
    try:
        rows = tomllib.loads(text).get("passage", [])
    except tomllib.TOMLDecodeError as exc:
        raise LabelFileError(str(exc)) from exc
    confirmed: dict[str, set[int]] = {}
    for number, row in enumerate(rows, start=1):
        verdict = row.get("verdict")
        if (
            verdict not in VERDICTS
            or "question" not in row
            or not isinstance(row.get("chunk"), int)
        ):
            raise LabelFileError(
                f"passage {number}: needs question, chunk and a verdict of {', '.join(VERDICTS)}"
            )
        if verdict == "yes":
            confirmed.setdefault(str(row["question"]), set()).add(row["chunk"])
    return {question: frozenset(chunks) for question, chunks in confirmed.items()}


async def unsearchable(session: AsyncSession, chunk_ids: Iterable[int]) -> set[int]:
    """The labelled chunks search can no longer return: gone, or replaced by a re-ingestion."""
    wanted = set(chunk_ids)
    current = await session.scalars(
        select(Chunk.id).where(Chunk.id.in_(wanted), Chunk.superseded_at.is_(None))
    )
    return wanted - set(current)


def fuse(vector: Sequence[int], keyword: Sequence[int], fusion: Fusion = CURRENT) -> list[int]:
    """Hybrid ranking as `search()` builds it, with the variant's full-text share."""
    fused = fuse_rankings(
        list(vector), list(keyword), fusion.keyword_candidates, fusion.keyword_weight
    )
    return [chunk_id for chunk_id, _ in fused]


def first_relevant(ranking: Sequence[int], relevant: frozenset[int]) -> int | None:
    return next((rank for rank, chunk in enumerate(ranking, start=1) if chunk in relevant), None)


def score(rankings: Sequence[Sequence[int]], relevant: Sequence[frozenset[int]]) -> Scores:
    """hit@1 and hit@5 count questions; Recall@5 is the share of a question's passages in its top
    five, averaged; MRR@5 the mean of 1 / rank of the first one, 0 when none is in the top five."""
    firsts = [
        first_relevant(ranking, gold) for ranking, gold in zip(rankings, relevant, strict=True)
    ]
    return Scores(
        questions=len(firsts),
        hit1=sum(first == 1 for first in firsts),
        hit5=sum(first is not None and first <= TOP for first in firsts),
        recall5=mean(
            len(gold & set(ranking[:TOP])) / len(gold)
            for ranking, gold in zip(rankings, relevant, strict=True)
        )
        if firsts
        else 0.0,
        mrr5=mean(1 / first if first is not None and first <= TOP else 0.0 for first in firsts)
        if firsts
        else 0.0,
    )


def moved(before: Sequence[int | None], after: Sequence[int | None]) -> tuple[int, int, int]:
    """How many questions' first relevant passage moved up, stayed, or moved down."""
    up = same = down = 0
    for old, new in zip(before, after, strict=True):
        old_rank, new_rank = old or 10**6, new or 10**6
        up += new_rank < old_rank
        same += new_rank == old_rank
        down += new_rank > old_rank
    return up, same, down


@dataclass(frozen=True)
class FusionVerdict:
    tuning: list[tuple[Fusion, float]]  # every variant's lenient MRR@5 on the tuning questions
    chosen: Fusion
    asked: tuple[float, float]  # current, chosen
    hand: tuple[float, float]
    asked_moved: tuple[int, int, int]
    adopted: bool


def judge_fusion(
    queries: Sequence[Query],
    vector: Mapping[str, Sequence[int]],
    keyword: Mapping[str, Sequence[int]],
    relevant: Mapping[str, frozenset[int]],
    current: Fusion = CURRENT,
    grid: Sequence[Fusion] = FUSION_GRID,
) -> FusionVerdict:
    """Tune on the passage questions that are not asked, then judge the choice once on the
    asked and hand-written ones. The choice is adopted only if it raises MRR@5 on the asked
    questions, with more of them up than down, and does not lower it on the hand-written ones."""

    def mrr(fusion: Fusion, subset: Sequence[Query]) -> float:
        rankings = [fuse(vector[query.id], keyword[query.id], fusion) for query in subset]
        return score(rankings, [relevant[query.id] for query in subset]).mrr5

    def firsts(fusion: Fusion, subset: Sequence[Query]) -> list[int | None]:
        return [
            first_relevant(fuse(vector[query.id], keyword[query.id], fusion), relevant[query.id])
            for query in subset
        ]

    tuning_set = [query for query in queries if PASSAGE in query.sets and ASKED not in query.sets]
    asked = [query for query in queries if ASKED in query.sets]
    hand = [query for query in queries if HAND in query.sets]
    tuning = [(fusion, mrr(fusion, tuning_set)) for fusion in grid]
    # Among equals the current setting stays; otherwise max() keeps the first in grid order.
    chosen = max(tuning, key=lambda pair: (pair[1], pair[0] == current))[0]
    asked_scores = (mrr(current, asked), mrr(chosen, asked))
    hand_scores = (mrr(current, hand), mrr(chosen, hand))
    asked_moved = moved(firsts(current, asked), firsts(chosen, asked))
    adopted = (
        chosen != current
        and asked_scores[1] > asked_scores[0]
        and asked_moved[0] > asked_moved[2]
        and hand_scores[1] >= hand_scores[0]
    )
    return FusionVerdict(tuning, chosen, asked_scores, hand_scores, asked_moved, adopted)

"""What practice earns: XP, a level, the streak and the coins.

Nothing here is stored. It is all worked out from the review history whenever it is asked
for, so a rule can change without touching the data. Each answer is judged on what had
happened by then, so later practice never changes what an earlier answer earned.

- **XP.** An answer earns ten times its score and five for answering. That is half as much
  again when the question was due for review, late or not, since clearing a backlog should
  still pay. Five more come in interview mode inside the limit. The day's first answer adds
  the streak's length that day, up to ten: the streak pays for coming back each day, not for
  answering more.
- **Levels** start at 50·(n−1)·n XP: Apprentice at 0, Journeyman at 100, Craftsman at 300, up
  to Daedalus at 2,100, about a hundred answers.
- **The streak** is the run of practice days with a graded answer. It holds through a day
  that has no answer yet, until that day ends.
- **Coins** are minted by the first answer that meets their condition. A condition about the
  library, such as every question answered once, counts the questions that existed when the
  answer was given, so a new question never takes a coin back. Retired questions are left
  out of the library, as they are out of practice.
"""

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import fsrs
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Attempt, Chunk, Grade, Question, QuestionSource, Review
from app.scheduling.schedule import AGAIN_BELOW, EASY_FROM, moment, review, scheduler

SCORE_XP = 10
ANSWERED_XP = 5
# Half as much again for a question that was due
DUE_BONUS = 0.5
INTERVIEW_XP = 5
STREAK_CAP = 10

LEVELS = (
    "Apprentice",
    "Journeyman",
    "Craftsman",
    "Inventor",
    "Architect",
    "Master Builder",
    "Daedalus",
)

STREAK_COIN_DAYS = 7
STRONG_TOPICS = 5
STRONG_MASTERY = 0.8
DUE_TO_CLEAR = 3
CONTRADICTED_TOO_MANY = 2

# Replays each question's memory state from its ratings. Fuzzing only moves due dates, so
# without it the state is the scheduler's own, and the same every time.
REPLAY = scheduler(fuzz=False)


@dataclass(frozen=True)
class Answer:
    """A graded answer as progress sees it: its review, with what its attempt and grade add."""

    review_id: int
    attempt_id: int
    question_id: int
    # The practice day it counted for, and the one its question was due again after it
    day: date
    due: date
    rating: int
    score: float
    contradicted: int
    seconds: float | None
    time_limit: int | None
    # When it was recorded
    at: datetime

    @property
    def inside_limit(self) -> bool:
        """Answered in interview mode, within the time it was given."""
        return (
            self.time_limit is not None
            and self.seconds is not None
            and self.seconds <= self.time_limit
        )


@dataclass(frozen=True)
class LibraryQuestion:
    """A question accepted into the library or retired from it, as progress needs it."""

    id: int
    topic_id: int | None
    accepted: bool
    created_at: datetime
    # The documents its passages come from
    documents: frozenset[int]


@dataclass(frozen=True)
class Xp:
    """What one answer earned, part by part."""

    score: int
    answered: int
    due: int
    interview: int
    streak: int

    @property
    def total(self) -> int:
        return self.score + self.answered + self.due + self.interview + self.streak


def whole(value: float) -> int:
    """Rounded half up. Rounding to six places first keeps 10 × 0.35 at 3.5, not 3.4999…"""
    return math.floor(round(value, 6) + 0.5)


def xp_for(answer: Answer, was_due: bool, streak: int, first_of_day: bool) -> Xp:
    score = whole(SCORE_XP * answer.score)
    return Xp(
        score=score,
        answered=ANSWERED_XP,
        due=whole(DUE_BONUS * (score + ANSWERED_XP)) if was_due else 0,
        interview=INTERVIEW_XP if answer.inside_limit else 0,
        streak=min(streak, STREAK_CAP) if first_of_day else 0,
    )


@dataclass(frozen=True)
class Level:
    number: int
    name: str
    # The XP it starts at, and where the next one starts and what it is called; None at the top
    start: int
    next_start: int | None
    next_name: str | None


def level_start(number: int) -> int:
    """The XP a level starts at: 0, 100, 300, 600, … Each takes 100 XP more than the last."""
    return 50 * (number - 1) * number


def level_for(xp: int) -> Level:
    number = max(n for n in range(1, len(LEVELS) + 1) if level_start(n) <= xp)
    if number == len(LEVELS):
        return Level(number, LEVELS[-1], level_start(number), None, None)
    return Level(
        number, LEVELS[number - 1], level_start(number), level_start(number + 1), LEVELS[number]
    )


def run_to(days: set[date], day: date) -> int:
    """How many practice days in a row end on `day`."""
    count = 0
    while day - timedelta(days=count) in days:
        count += 1
    return count


@dataclass(frozen=True)
class Streak:
    # The current run of days: it holds through today while today has no answer yet
    days: int
    today: bool
    best: int


def streak_on(days: set[date], today: date) -> Streak:
    current = run_to(days, today) or run_to(days, today - timedelta(days=1))
    best = max((run_to(days, day) for day in days), default=0)
    return Streak(current, today in days, best)


@dataclass(frozen=True)
class Coin:
    id: str
    name: str
    glyph: str
    # What mints it, as the shelf says it
    condition: str


COINS = (
    Coin("first-thread", "First thread", "α", "your first graded answer"),
    Coin("theseus", "Theseus", "Θ", f"a {STREAK_COIN_DAYS}-day streak"),
    Coin(
        "minotaur-slayer",
        "Minotaur slayer",
        "Μ",
        f"{EASY_FROM} or more on a question you once scored under {AGAIN_BELOW}",
    ),
    Coin("cartographer", "Cartographer", "Χ", "a question from every source"),
    Coin("knossos", "Knossos", "Κ", f"{STRONG_TOPICS} topics at {STRONG_MASTERY} mastery or more"),
    Coin("hermes", "Hermes", "Η", f"{EASY_FROM} or more in interview mode, inside the limit"),
    Coin("ariadne", "Ariadne", "Α", f"every due review cleared in a day (at least {DUE_TO_CLEAR})"),
    Coin("icarus", "Icarus", "Ι", "two or more contradicted claims in one answer"),
    Coin("labyrinth-walker", "Labyrinth walker", "Λ", "every question answered once"),
    Coin("daedalus", "Daedalus", "Δ", "reach the top level"),
)


@dataclass(frozen=True)
class Step:
    """One answer in the history: what it earned, and where it left practice."""

    answer: Answer
    xp: Xp
    # XP after it, and the streak on its day
    total: int
    streak: int
    # The coins it minted
    minted: tuple[Coin, ...]

    @property
    def level(self) -> Level:
        return level_for(self.total)

    @property
    def level_up(self) -> bool:
        return level_for(self.total - self.xp.total).number < self.level.number


@dataclass
class Ledger:
    """Practice as it stands after the answers added so far, one answer at a time."""

    questions: dict[int, LibraryQuestion]
    answered: dict[int, list[Answer]] = field(default_factory=dict)
    cards: dict[int, fsrs.Card] = field(default_factory=dict)
    days: set[date] = field(default_factory=set)
    total: int = 0
    steps: list[Step] = field(default_factory=list)
    minted: dict[str, Step] = field(default_factory=dict)

    def library(self, by: datetime | None = None) -> list[LibraryQuestion]:
        """The questions in the library, or those it held at a moment."""
        return [
            question
            for question in self.questions.values()
            if question.accepted and (by is None or question.created_at <= by)
        ]

    def topic_mastery(self, library: Iterable[LibraryQuestion], day: date) -> dict[int, float]:
        """Each topic's mastery on a day, as the dashboard shows it (`app.scheduling.mastery`):
        the mean over its questions of the latest score times the chance of recalling it."""
        grouped: dict[int, list[float]] = {}
        for question in library:
            if question.topic_id is None:
                continue
            answers = self.answered.get(question.id)
            mastery = (
                answers[-1].score
                * REPLAY.get_card_retrievability(self.cards[question.id], moment(day))
                if answers
                else 0.0
            )
            grouped.setdefault(question.topic_id, []).append(mastery)
        return {topic: sum(values) / len(values) for topic, values in grouped.items()}

    def strong_topics(self, library: Iterable[LibraryQuestion], day: date) -> int:
        return sum(
            mastery >= STRONG_MASTERY for mastery in self.topic_mastery(library, day).values()
        )

    def sources(self, library: Iterable[LibraryQuestion]) -> tuple[int, int]:
        """The documents behind the library's questions: how many an answer has come from,
        out of how many there are."""
        everything = {document for question in library for document in question.documents}
        covered = {
            document
            for question_id in self.answered
            if question_id in self.questions
            for document in self.questions[question_id].documents
        }
        return len(everything & covered), len(everything)

    def add(self, answer: Answer) -> Step:
        earlier = self.answered.get(answer.question_id, [])
        was_due = bool(earlier) and earlier[-1].due <= answer.day
        first_of_day = answer.day not in self.days
        self.days.add(answer.day)
        streak = run_to(self.days, answer.day)
        xp = xp_for(answer, was_due, streak, first_of_day)
        self.total += xp.total
        self.answered[answer.question_id] = [*earlier, answer]
        self.cards[answer.question_id] = review(
            self.cards.get(answer.question_id) or fsrs.Card(card_id=answer.question_id),
            fsrs.Rating(answer.rating),
            answer.day,
            REPLAY,
        )
        snapshot = Snapshot(self, answer, earlier, streak, self.library(by=answer.at))
        minted = tuple(
            coin for coin in COINS if coin.id not in self.minted and MINTS[coin.id](snapshot)
        )
        step = Step(answer, xp, self.total, streak, minted)
        self.steps.append(step)
        for coin in minted:
            self.minted[coin.id] = step
        return step


@dataclass(frozen=True)
class Snapshot:
    """Practice just after an answer, as the coins' conditions judge it."""

    ledger: Ledger
    answer: Answer
    # The same question's earlier answers
    earlier: list[Answer]
    streak: int
    # The library as it stood when the answer was given
    library: list[LibraryQuestion]


def cleared_every_due_review(snapshot: Snapshot) -> bool:
    """Every question due when the answer's day began has been answered that day, and there
    were at least three."""
    day = snapshot.answer.day
    due = []
    for question in snapshot.library:
        before = [a for a in snapshot.ledger.answered.get(question.id, []) if a.day < day]
        if before and before[-1].due <= day:
            due.append(question.id)
    return len(due) >= DUE_TO_CLEAR and all(
        any(a.day == day for a in snapshot.ledger.answered[question_id]) for question_id in due
    )


def from_every_source(snapshot: Snapshot) -> bool:
    covered, everything = snapshot.ledger.sources(snapshot.library)
    return everything > 0 and covered == everything


MINTS: dict[str, Callable[[Snapshot], bool]] = {
    "first-thread": lambda s: True,
    "theseus": lambda s: s.streak >= STREAK_COIN_DAYS,
    "minotaur-slayer": lambda s: (
        s.answer.score >= EASY_FROM and any(a.score < AGAIN_BELOW for a in s.earlier)
    ),
    "cartographer": from_every_source,
    "knossos": lambda s: s.ledger.strong_topics(s.library, s.answer.day) >= STRONG_TOPICS,
    "hermes": lambda s: s.answer.inside_limit and s.answer.score >= EASY_FROM,
    "ariadne": cleared_every_due_review,
    "icarus": lambda s: s.answer.contradicted >= CONTRADICTED_TOO_MANY,
    "labyrinth-walker": lambda s: (
        bool(s.library) and all(question.id in s.ledger.answered for question in s.library)
    ),
    "daedalus": lambda s: s.ledger.total >= level_start(len(LEVELS)),
}


def walk(answers: Sequence[Answer], questions: Iterable[LibraryQuestion]) -> Ledger:
    """The history, answer by answer, in the order the answers were recorded."""
    ledger = Ledger({question.id: question for question in questions})
    for answer in answers:
        ledger.add(answer)
    return ledger


@dataclass(frozen=True)
class CoinState:
    coin: Coin
    # The practice day it was minted; None while it is still to earn
    minted_on: date | None
    # How far along a coin still to earn is, where that can be counted
    have: int | None = None
    need: int | None = None


@dataclass(frozen=True)
class Progress:
    xp: int
    level: Level
    streak: Streak
    answers: int
    coins: list[CoinState]


def progress(ledger: Ledger, today: date) -> Progress:
    streak = streak_on(ledger.days, today)
    library = ledger.library()
    covered, sources = ledger.sources(library)
    counts: dict[str, tuple[int, int]] = {
        "first-thread": (len(ledger.steps), 1),
        "theseus": (streak.days, STREAK_COIN_DAYS),
        "cartographer": (covered, sources),
        "knossos": (ledger.strong_topics(library, today), STRONG_TOPICS),
        "labyrinth-walker": (
            sum(question.id in ledger.answered for question in library),
            len(library),
        ),
        "daedalus": (ledger.total, level_start(len(LEVELS))),
    }
    coins = []
    for coin in COINS:
        if coin.id in ledger.minted:
            coins.append(CoinState(coin, ledger.minted[coin.id].answer.day))
        else:
            have, need = counts.get(coin.id, (None, None))
            coins.append(CoinState(coin, None, have, need))
    return Progress(ledger.total, level_for(ledger.total), streak, len(ledger.steps), coins)


async def load(session: AsyncSession) -> tuple[list[Answer], list[LibraryQuestion]]:
    """Every graded answer in the order it was recorded, and the questions in the library or
    retired from it."""
    rows = await session.execute(
        select(
            Review.id.label("review_id"),
            Review.attempt_id,
            Review.question_id,
            Review.day,
            Review.due,
            Review.rating,
            Review.score,
            Grade.contradicted,
            Attempt.seconds,
            Attempt.time_limit,
            Review.created_at.label("at"),
        )
        .join(Attempt, Attempt.id == Review.attempt_id)
        .join(Grade, Grade.id == Review.grade_id)
        .order_by(Review.id)
    )
    # Only a successful grade is reviewed, and it always counts its contradicted claims.
    answers = [Answer(**row) for row in rows.mappings()]
    documents: dict[int, set[int]] = {}
    for question_id, document_id in (
        await session.execute(
            select(QuestionSource.question_id, Chunk.document_id).join(
                Chunk, Chunk.id == QuestionSource.chunk_id
            )
        )
    ).tuples():
        documents.setdefault(question_id, set()).add(document_id)
    questions = [
        LibraryQuestion(
            id=question_id,
            topic_id=topic_id,
            accepted=status == "accepted",
            created_at=created_at,
            documents=frozenset(documents.get(question_id, ())),
        )
        for question_id, topic_id, status, created_at in (
            await session.execute(
                select(Question.id, Question.topic_id, Question.status, Question.created_at).where(
                    Question.status.in_(("accepted", "retired"))
                )
            )
        ).tuples()
    ]
    return answers, questions

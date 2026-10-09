"""Mock interviews: a few library questions asked in turn, each answer perhaps followed up on
what it missed, then a report (`app.interview`).

An interview starts with its questions picked the way practice picks them, and asks the first.
Each answer moves it on: graded, then followed up when the grade shows a gap, then the next
question, until the last is answered or the interview is ended. An answer to a library question
is an ordinary attempt, graded, scheduled and earning XP as in practice; a follow-up's answer
is graded against the follow-up's own key points and kept on the interview.

Every grade an interview asks for is held to the daily limits like any other: an answer past a
limit is refused with 429 before anything is written, and a follow-up is only written while a
grade is left for its answer. Interviews are their user's own, and go with their practice.
"""

from datetime import UTC, date, datetime, timedelta
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.grading import (
    ATTEMPT_PARTS,
    REFUSED,
    AnswerIn,
    AttemptOut,
    ClaimOut,
    GraderDep,
    KeyPointGradeOut,
    _attempts_out,
    cited_chunks,
    claims_out,
    key_points_out,
    save_answer,
)
from app.api.users import UserDep
from app.core.config import Settings, get_settings
from app.db.models import Attempt, Interview, InterviewTurn, Question, Topic
from app.db.session import get_session
from app.grading.grader import spent_today
from app.grading.limits import LimitReached, allowance, refusal_message, take
from app.interview import places
from app.interview.follow_ups import FollowUpWriter, Writer
from app.interview.picking import plan
from app.interview.report import Round, report
from app.llm.models import follow_up_model
from app.scheduling.mastery import standings
from app.scheduling.schedule import practice_day, rating_name

router = APIRouter(tags=["interviews"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]

# Seconds an answer has, as in practice's interview mode
TIME_LIMIT = 180
# Longer than any request runs on the demo's host (300 s): an interview marked as moving on for
# longer belongs to a request that failed, and the next look carries it on
STALLED = timedelta(minutes=5)


async def get_writer(request: Request, session: SessionDep, settings: SettingsDep) -> Writer | None:
    """Who writes follow-ups, or None when no model can. Like the grader, the writer is built
    once a process, with its pacer starting from what the day has already spent."""
    writer = getattr(request.app.state, "writer", None)
    if writer is None:
        model = follow_up_model(settings, await spent_today(session))
        if model is None:
            return None
        writer = request.app.state.writer = FollowUpWriter(model)
    return writer


WriterDep = Annotated[Writer | None, Depends(get_writer)]


class InterviewIn(BaseModel):
    # How many library questions to ask
    size: Literal[3, 5] = 3
    # Keep to one topic; null to ask across topics
    topic_id: int | None = None


class InterviewAnswerIn(AnswerIn):
    # The turn answered: the one the interview is waiting on
    turn_id: int


class AimOut(BaseModel):
    kind: Literal["contradicted", "missing", "partial"]
    # The claim a passage contradicts, or the key point the answer missed or only partly made
    text: str


class FollowUpGradeOut(BaseModel):
    status: str
    error: str | None
    grader_model: str | None
    score: float | None
    coverage: float | None
    contradicted: int | None
    clarity: int | None
    key_points: list[KeyPointGradeOut]
    claims: list[ClaimOut]
    strengths: list[str]
    gaps: list[str]
    errors: list[str]
    improved_answer: str | None
    seconds: float | None


class TurnOut(BaseModel):
    id: int
    # Which of the interview's questions it belongs to, from 0
    round: int
    kind: Literal["question", "follow_up"]
    # The library question; for a follow-up, the one it follows
    question_id: int
    # What was asked: the library question, or the follow-up's own words
    text: str
    topic: str | None
    # Seconds an answer has
    time_limit: int
    # The answer to a library question, graded and scheduled as in practice; null until it is
    # answered
    attempt: AttemptOut | None
    # Why no follow-up came after the answer, when none did
    no_follow_up: str | None
    # A follow-up's aim, and its answer and grade once it is answered
    aim: AimOut | None
    answer: str | None
    seconds: float | None
    grade: FollowUpGradeOut | None


class TopicOut(BaseModel):
    id: int | None
    name: str | None


class RoundOut(BaseModel):
    question_id: int
    question: str
    topic: str | None
    # The answer's score, the rating it earned and the day the question comes back
    score: float | None
    rating: Literal["again", "hard", "good", "easy"] | None
    due: date | None
    # The follow-up's score, and whether it made up for the gap; null without one graded
    follow_up_score: float | None
    recovered: bool | None


class ReportOut(BaseModel):
    # Library questions answered, and their mean score over those graded
    answered: int
    mean_score: float | None
    # Follow-ups answered and graded, and those that made up for the gap
    follow_ups: int
    recovered: int
    # The topics to go back to, in the order they came up
    review: list[TopicOut]
    rounds: list[RoundOut]


class InterviewOut(BaseModel):
    id: int
    status: Literal["asking", "finished", "ended"]
    # How many library questions it asks
    size: int
    topic: TopicOut | None
    created_at: datetime
    finished_at: datetime | None
    # Every question put so far, in order
    turns: list[TurnOut]
    # The turn waiting for an answer; null once the interview is over
    waiting: int | None
    # Once it is over
    report: ReportOut | None


class InterviewSummaryOut(BaseModel):
    id: int
    status: Literal["asking", "finished", "ended"]
    size: int
    topic: TopicOut | None
    created_at: datetime
    finished_at: datetime | None
    answered: int


def _follow_up_grade_out(turn: InterviewTurn, cited: dict) -> FollowUpGradeOut | None:
    found: dict[str, Any] | None = turn.grade
    if found is None:
        return None
    return FollowUpGradeOut(
        status=found["status"],
        error=found.get("error"),
        grader_model=found.get("grader_model"),
        score=found.get("score"),
        coverage=found.get("coverage"),
        contradicted=found.get("contradicted"),
        clarity=found.get("clarity"),
        key_points=key_points_out(found.get("key_points", []), turn.key_points or []),
        claims=claims_out(found.get("claims", []), cited),
        strengths=found.get("strengths", []),
        gaps=found.get("gaps", []),
        errors=found.get("errors", []),
        improved_answer=found.get("improved_answer"),
        seconds=found.get("seconds"),
    )


def waiting_turn(interview: Interview) -> InterviewTurn | None:
    """The turn the interview waits on: the last one put, while it is unanswered."""
    if interview.status != "asking" or not interview.turns:
        return None
    last = interview.turns[-1]
    answered = last.attempt_id is not None if last.kind == "question" else last.answer is not None
    return None if answered else last


async def _interview(
    session: AsyncSession, user_id: int, interview_id: int, *, lock: bool = False
) -> Interview:
    """One of the user's interviews, with its turns read afresh; anyone else's is not found.
    Locked, it holds back any other request for it until the transaction ends."""
    found = select(Interview).where(Interview.id == interview_id, Interview.user_id == user_id)
    if lock:
        found = found.with_for_update()
    interview = await session.scalar(found.execution_options(populate_existing=True))
    if interview is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "interview not found")
    await session.refresh(interview, ["turns"])
    return interview


async def _topics(session: AsyncSession, ids: set[int | None]) -> dict[int | None, str | None]:
    known = {topic_id for topic_id in ids if topic_id is not None}
    names: dict[int | None, str | None] = {None: None}
    if known:
        rows = await session.execute(select(Topic.id, Topic.name).where(Topic.id.in_(known)))
        names |= dict(rows.tuples().all())
    return names


async def _interview_out(session: AsyncSession, user_id: int, interview: Interview) -> InterviewOut:
    turns = interview.turns
    questions = {
        question.id: question
        for question in await session.scalars(
            select(Question).where(Question.id.in_({turn.question_id for turn in turns}))
        )
    }
    attempt_ids = [turn.attempt_id for turn in turns if turn.attempt_id is not None]
    attempts = list(
        await session.scalars(
            select(Attempt)
            .options(*ATTEMPT_PARTS)
            .where(Attempt.id.in_(attempt_ids), Attempt.user_id == user_id)
            .execution_options(populate_existing=True)
        )
    )
    shown = {
        attempt.id: out
        for attempt, out in zip(
            attempts, await _attempts_out(session, user_id, attempts), strict=True
        )
    }
    reviews = {attempt.id: attempt.review for attempt in attempts}
    cited = await cited_chunks(session, [(turn.grade or {}).get("claims", []) for turn in turns])
    topic_names = await _topics(
        session, {question.topic_id for question in questions.values()} | {interview.topic_id}
    )

    def topic_of(question_id: int) -> str | None:
        question = questions.get(question_id)
        return topic_names.get(question.topic_id) if question else None

    turns_out = []
    for turn in turns:
        follow_up = turn.kind == "follow_up"
        aim = turn.aim or {}
        turns_out.append(
            TurnOut(
                id=turn.id,
                round=turn.round,
                kind="follow_up" if follow_up else "question",
                question_id=turn.question_id,
                text=turn.text if follow_up and turn.text else questions[turn.question_id].text,
                topic=topic_of(turn.question_id),
                time_limit=TIME_LIMIT,
                attempt=shown.get(turn.attempt_id) if turn.attempt_id is not None else None,
                no_follow_up=turn.no_follow_up,
                aim=AimOut(kind=aim["kind"], text=aim["text"]) if aim else None,
                answer=turn.answer,
                seconds=turn.seconds,
                grade=_follow_up_grade_out(turn, cited),
            )
        )

    waiting = waiting_turn(interview)
    summary = None
    if interview.status != "asking":
        rounds = []
        for turn in turns:
            if turn.kind != "question":
                continue
            attempt = next((a for a in attempts if a.id == turn.attempt_id), None)
            grades = [
                grade for grade in (attempt.grades if attempt else []) if grade.score is not None
            ]
            review = reviews.get(turn.attempt_id) if turn.attempt_id is not None else None
            follow = next(
                (t for t in turns if t.round == turn.round and t.kind == "follow_up"), None
            )
            follow_score = (follow.grade or {}).get("score") if follow else None
            question = questions[turn.question_id]
            rounds.append(
                (
                    question,
                    Round(
                        question_id=question.id,
                        topic_id=question.topic_id,
                        answered=attempt is not None,
                        score=grades[0].score if grades else None,
                        follow_up_score=follow_score,
                        rating=rating_name(review.rating) if review else None,
                        due=review.due if review else None,
                    ),
                )
            )
        found = report([round_ for _, round_ in rounds])
        summary = ReportOut(
            answered=found.answered,
            mean_score=found.mean_score,
            follow_ups=found.follow_ups,
            recovered=found.recovered,
            review=[
                TopicOut(id=topic_id, name=topic_names.get(topic_id)) for topic_id in found.review
            ],
            rounds=[
                RoundOut(
                    question_id=round_.question_id,
                    question=question.text,
                    topic=topic_names.get(round_.topic_id),
                    score=round_.score,
                    rating=round_.rating,
                    due=round_.due,
                    follow_up_score=round_.follow_up_score,
                    recovered=round_.recovered,
                )
                for question, round_ in rounds
            ],
        )

    return InterviewOut(
        id=interview.id,
        status=interview.status,
        size=len(interview.questions),
        topic=TopicOut(id=interview.topic_id, name=topic_names.get(interview.topic_id))
        if interview.topic_id is not None
        else None,
        created_at=interview.created_at,
        finished_at=interview.finished_at,
        turns=turns_out,
        waiting=waiting.id if waiting else None,
        report=summary,
    )


async def _move_on(
    session: AsyncSession,
    settings: Settings,
    user_id: int,
    grader: Any,
    writer: Writer | None,
    interview: Interview,
    answered: dict[str, int] | None = None,
) -> None:
    """Run the interview's graph on, marked as moving on (the caller has set the mark and
    committed), then clear the mark, and forget the graph's place once the interview is over."""
    from app.interview.graph import Context, move_on

    context = Context(session, settings, user_id, grader, writer)
    interview_id = interview.id
    try:
        over = await move_on(context, interview, answered)
    finally:
        # Whatever a failed step left unfinished is undone first.
        await session.rollback()
        interview = await session.get_one(Interview, interview_id, populate_existing=True)
        interview.moving_since = None
        await session.commit()
    if over:
        await places.forget(session, [interview_id])
        await session.commit()


@router.post("/interviews", status_code=status.HTTP_201_CREATED, responses=REFUSED)
async def start_interview(
    body: InterviewIn,
    session: SessionDep,
    settings: SettingsDep,
    user_id: UserDep,
    grader: GraderDep,
    writer: WriterDep,
) -> InterviewOut:
    """Start a mock interview, with its questions picked the way practice picks them, and ask
    the first. Refused (429) while the daily limits leave no grade."""
    now = datetime.now(UTC)
    refused = (await allowance(session, settings, user_id, now)).refusal()
    if refused is not None:
        raise LimitReached(refused, refusal_message(refused, settings))
    if body.topic_id is not None and await session.get(Topic, body.topic_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "topic not found")
    day = practice_day(now, settings.practice_zone)
    picked = plan(await standings(session, user_id, day), day, body.size, body.topic_id)
    if not picked:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "no questions to ask yet; generate some first"
        )
    interview = Interview(
        user_id=user_id, questions=picked, topic_id=body.topic_id, moving_since=now
    )
    session.add(interview)
    await session.commit()
    await _move_on(session, settings, user_id, grader, writer, interview)
    return await _interview_out(session, user_id, await _interview(session, user_id, interview.id))


@router.get("/interviews")
async def list_interviews(session: SessionDep, user_id: UserDep) -> list[InterviewSummaryOut]:
    """Your interviews, newest first."""
    interviews = list(
        await session.scalars(
            select(Interview)
            .where(Interview.user_id == user_id)
            .order_by(Interview.created_at.desc(), Interview.id.desc())
        )
    )
    answered: dict[int, int] = {}
    for interview_id, attempt_id in await session.execute(
        select(InterviewTurn.interview_id, InterviewTurn.attempt_id).where(
            InterviewTurn.interview_id.in_([interview.id for interview in interviews]),
            InterviewTurn.kind == "question",
        )
    ):
        answered[interview_id] = answered.get(interview_id, 0) + (attempt_id is not None)
    names = await _topics(session, {interview.topic_id for interview in interviews})
    return [
        InterviewSummaryOut(
            id=interview.id,
            status=interview.status,
            size=len(interview.questions),
            topic=TopicOut(id=interview.topic_id, name=names.get(interview.topic_id))
            if interview.topic_id is not None
            else None,
            created_at=interview.created_at,
            finished_at=interview.finished_at,
            answered=answered.get(interview.id, 0),
        )
        for interview in interviews
    ]


@router.get("/interviews/{interview_id}")
async def get_interview(
    interview_id: int,
    session: SessionDep,
    settings: SettingsDep,
    user_id: UserDep,
    grader: GraderDep,
    writer: WriterDep,
) -> InterviewOut:
    """One of your interviews: every question put, each answer's grade, and the report once it
    is over. One a failed request left halfway is carried on first."""
    interview = await _interview(session, user_id, interview_id)
    if stalled(interview):
        interview = await _interview(session, user_id, interview_id, lock=True)
        if stalled(interview):
            interview.moving_since = datetime.now(UTC)
            await session.commit()
            await _move_on(session, settings, user_id, grader, writer, interview)
        interview = await _interview(session, user_id, interview_id)
    return await _interview_out(session, user_id, interview)


def stalled(interview: Interview) -> bool:
    """Whether the interview neither waits for an answer nor is being moved on: a request that
    was moving it on failed."""
    if interview.status != "asking" or waiting_turn(interview) is not None:
        return False
    since = interview.moving_since
    return since is None or datetime.now(UTC) - since > STALLED


@router.post("/interviews/{interview_id}/answers", responses=REFUSED)
async def answer_interview(
    interview_id: int,
    body: InterviewAnswerIn,
    session: SessionDep,
    settings: SettingsDep,
    user_id: UserDep,
    grader: GraderDep,
    writer: WriterDep,
) -> InterviewOut:
    """Answer the question the interview waits on. The answer is graded, followed up when its
    grade shows a gap, and the interview moves on to its next question or its end. Past a
    daily limit nothing is written down and the answer is refused (429)."""
    interview = await _interview(session, user_id, interview_id, lock=True)
    if interview.status != "asking":
        raise HTTPException(status.HTTP_409_CONFLICT, "the interview is over")
    turn = waiting_turn(interview)
    if turn is None or turn.id != body.turn_id:
        raise HTTPException(status.HTTP_409_CONFLICT, "that question is not waiting for an answer")
    if turn.kind == "question":
        attempt, slot = await save_answer(session, settings, user_id, turn.question_id, body)
        turn.attempt_id = attempt.id
    else:
        slot = await take(session, settings, user_id, datetime.now(UTC))
        turn.answer = body.answer
        turn.seconds = body.seconds
        turn.answered_at = datetime.now(UTC)
    # The answer and its slot are kept whatever becomes of its grading.
    interview.moving_since = datetime.now(UTC)
    await session.commit()
    await _move_on(
        session, settings, user_id, grader, writer, interview, {"turn": turn.id, "slot": slot.id}
    )
    return await _interview_out(session, user_id, await _interview(session, user_id, interview_id))


@router.post("/interviews/{interview_id}/end")
async def end_interview(interview_id: int, session: SessionDep, user_id: UserDep) -> InterviewOut:
    """End an interview before its last question. The report covers what was answered."""
    interview = await _interview(session, user_id, interview_id, lock=True)
    moving = interview.moving_since is not None and not stalled(interview)
    if interview.status == "asking" and moving and waiting_turn(interview) is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "the interview is moving on; end it once it has"
        )
    if interview.status == "asking":
        interview.status = "ended"
        interview.finished_at = datetime.now(UTC)
        await places.forget(session, [interview.id])
        await session.commit()
    return await _interview_out(session, user_id, await _interview(session, user_id, interview_id))

"""The interviewer: a LangGraph graph that asks an interview's questions in turn.

Each request moves an interview on, then stops where it waits for the candidate's next answer:

    ask ──► wait ──► grade ──► follow_up ──► wait (a follow-up was written)
     ▲                  │          │
     │                  ▼          ▼
     └──────────────── advance ◄───┘ (none) ──► finish (no question left)

`grade` sends a library question's answer on to `follow_up`, and a follow-up's to `advance`.

Its place is kept by LangGraph's Postgres saver in the app's own database, as the thread
"interview-<id>", so the next request carries on from it in whatever process serves it. The
state holds ids alone. Whatever resumes a graph is kept with its place, so an answer is written
to `attempts` or `interview_turns` first and the graph is resumed with the turn's and the
grade slot's ids. Once the interview is over its thread is deleted.

A step paused at `wait` runs again from its start when the graph resumes, and so does a step a
failed request cut short, so each step reads what is already written before it writes.

LangGraph is imported here and nowhere else at start-up: with langchain-core it adds about a
third of a second, which only an interview should pay. Its LangSmith tracing is switched off
for every run, whatever the environment says; model calls are traced to Langfuse as any are.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypedDict

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from langsmith import tracing_context
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from pydantic_ai.models import Model
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.grading import grade_and_schedule
from app.core.config import Settings
from app.db.models import Attempt, Grade, GradeRequest, Interview, InterviewTurn, Question
from app.grading import limits
from app.grading.grader import PROMPT_VERSION, Outcome, question_sources, try_grading
from app.interview import routing
from app.interview.follow_ups import Writer
from app.interview.places import thread
from app.questions import batch


class State(TypedDict, total=False):
    interview_id: int
    # How many questions it asks, and which of them is being asked, from 0
    size: int
    round: int
    # The turn asked last, and what kind it is
    turn: int
    kind: str
    # Whether a follow-up was written for the round
    followed_up: bool
    # The grade slot taken for the answer the graph was resumed with
    slot: int


@dataclass(frozen=True)
class Context:
    """What the steps work with, given to each run and never kept with its place."""

    session: AsyncSession
    settings: Settings
    user_id: int
    grader: Model
    writer: Writer | None


async def _turn(session: AsyncSession, interview_id: int, round_: int, kind: str):
    return await session.scalar(
        select(InterviewTurn).where(
            InterviewTurn.interview_id == interview_id,
            InterviewTurn.round == round_,
            InterviewTurn.kind == kind,
        )
    )


async def ask(state: State, runtime: Runtime[Context]) -> State:
    """Put the round's library question."""
    session = runtime.context.session
    round_ = state["round"]
    turn = await _turn(session, state["interview_id"], round_, "question")
    if turn is None:
        interview = await session.get_one(Interview, state["interview_id"])
        turn = InterviewTurn(
            interview_id=interview.id,
            round=round_,
            kind="question",
            question_id=interview.questions[round_],
        )
        session.add(turn)
        await session.commit()
    return {"turn": turn.id, "kind": "question", "followed_up": False}


def wait(state: State) -> State:
    """Wait for the answer to the turn asked last. Nothing runs before the pause: the step runs
    again from here when the graph resumes."""
    answered = interrupt({"turn": state["turn"]})
    return {"slot": answered["slot"]}


async def grade(state: State, runtime: Runtime[Context]) -> State:
    """Grade the answer just given, in the slot taken for it."""
    context = runtime.context
    session = context.session
    turn = await session.get_one(InterviewTurn, state["turn"])
    slot = await session.get(GradeRequest, state["slot"])
    if turn.kind == "question":
        if turn.attempt_id is not None:
            graded = await session.scalar(
                select(Grade.id).where(Grade.attempt_id == turn.attempt_id)
            )
            if graded is None:
                attempt = await session.get_one(Attempt, turn.attempt_id)
                await grade_and_schedule(session, context.grader, context.settings, attempt, slot)
    elif turn.grade is None and turn.answer is not None:
        await grade_follow_up(session, context, turn, slot)
    return {}


async def grade_follow_up(
    session: AsyncSession, context: Context, turn: InterviewTurn, slot: GradeRequest | None
) -> None:
    """Grade a follow-up's answer against its own key points and passages, as a library
    question's would be, and keep the grade on the turn."""
    assert turn.text is not None and turn.answer is not None
    sources = await batch.sources_for(session, list(turn.chunk_ids or []))
    outcome = await try_grading(
        context.grader,
        turn.text,
        turn.key_points or [],
        sources,
        turn.answer,
        {"interview": turn.interview_id, "turn": turn.id},
    )
    turn.grade = follow_up_grade(outcome)
    await limits.charge(session, slot, context.user_id, None, outcome.answered_by, outcome.spent)
    await session.commit()


def follow_up_grade(outcome: Outcome) -> dict:
    """A follow-up's grade, as `Grade` records one."""
    graded = outcome.graded
    if graded is None:
        return {
            "status": "failed",
            "error": outcome.error,
            "prompt_version": PROMPT_VERSION,
            "usage": limits.usage_of(outcome.spent) if outcome.spent.requests else {},
        }
    return {
        "status": "graded",
        "grader_model": graded.model,
        "prompt_version": PROMPT_VERSION,
        "key_points": graded.key_points,
        "claims": graded.claims,
        "clarity": graded.clarity,
        "strengths": graded.strengths,
        "gaps": graded.gaps,
        "errors": graded.errors,
        "improved_answer": graded.improved_answer,
        "coverage": graded.result.coverage,
        "contradicted": graded.result.contradicted,
        "score": graded.result.score,
        "usage": graded.usage,
        "seconds": graded.seconds,
    }


async def follow_up(state: State, runtime: Runtime[Context]) -> State:
    """Write a follow-up about the gap in the answer, or note why there is none."""
    context = runtime.context
    session = context.session
    existing = await _turn(session, state["interview_id"], state["round"], "follow_up")
    if existing is not None:
        return {"turn": existing.id, "kind": "follow_up", "followed_up": True}
    turn = await session.get_one(InterviewTurn, state["turn"])
    if turn.no_follow_up is not None:
        return {"followed_up": False}

    question = await session.get_one(Question, turn.question_id)
    grade_ = None
    attempt = None
    if turn.attempt_id is not None:
        attempt = await session.get_one(Attempt, turn.attempt_id)
        grade_ = await session.scalar(
            select(Grade).where(Grade.attempt_id == attempt.id).order_by(Grade.id.desc()).limit(1)
        )
    gap = routing.aim(grade_, question.key_points)
    now = datetime.now(UTC)
    if isinstance(gap, str):
        why_not = gap
    elif (await limits.allowance(session, context.settings, context.user_id, now)).refusal():
        why_not = routing.NO_GRADE_LEFT
    elif context.writer is None:
        why_not = routing.NO_WRITER
    else:
        assert attempt is not None
        sources = await question_sources(session, question.id)
        written = await context.writer(question.text, sources, gap, attempt.answer)
        new = written.follow_up
        if new is not None:
            follow = InterviewTurn(
                interview_id=turn.interview_id,
                round=turn.round,
                kind="follow_up",
                question_id=question.id,
                text=new.text,
                key_points=new.key_points,
                chunk_ids=new.chunk_ids,
                aim=gap.as_aim(),
                writer_model=written.model,
                prompt_version=written.prompt_version,
                usage=written.usage,
            )
            session.add(follow)
            await session.commit()
            return {"turn": follow.id, "kind": "follow_up", "followed_up": True}
        why_not = routing.NOT_WRITTEN
        # What trying cost is kept on the question's turn, where the pacer counts it.
        turn.writer_model = written.model
        turn.prompt_version = written.prompt_version
        turn.usage = written.usage
    turn.no_follow_up = why_not
    await session.commit()
    return {"followed_up": False}


def advance(state: State) -> State:
    return {"round": state["round"] + 1}


async def finish(state: State, runtime: Runtime[Context]) -> State:
    session = runtime.context.session
    interview = await session.get_one(Interview, state["interview_id"])
    if interview.status == "asking":
        interview.status = "finished"
        interview.finished_at = datetime.now(UTC)
        await session.commit()
    return {}


def after_grade(state: State) -> str:
    return "follow_up" if state["kind"] == "question" else "advance"


def after_follow_up(state: State) -> str:
    return "wait" if state["followed_up"] else "advance"


def after_advance(state: State) -> str:
    return "ask" if state["round"] < state["size"] else "finish"


def build() -> StateGraph:
    graph = StateGraph(State, context_schema=Context)
    graph.add_node("ask", ask)
    graph.add_node("wait", wait)
    graph.add_node("grade", grade)
    graph.add_node("follow_up", follow_up)
    graph.add_node("advance", advance)
    graph.add_node("finish", finish)
    graph.add_edge(START, "ask")
    graph.add_edge("ask", "wait")
    graph.add_edge("wait", "grade")
    graph.add_conditional_edges("grade", after_grade, ["follow_up", "advance"])
    graph.add_conditional_edges("follow_up", after_follow_up, ["wait", "advance"])
    graph.add_conditional_edges("advance", after_advance, ["ask", "finish"])
    graph.add_edge("finish", END)
    return graph


@asynccontextmanager
async def saver_for(session: AsyncSession) -> AsyncIterator[AsyncPostgresSaver]:
    """LangGraph's saver, on a connection of its own to the session's database. It prepares no
    statements, so it works through a pooler such as Neon's as well as directly."""
    assert session.bind is not None
    url = session.bind.url.set(drivername="postgresql").render_as_string(hide_password=False)
    async with await AsyncConnection.connect(
        url, autocommit=True, prepare_threshold=None, row_factory=dict_row
    ) as connection:
        yield AsyncPostgresSaver(connection)


async def move_on(context: Context, interview: Interview, answered: dict | None = None) -> bool:
    """Run the interview until it waits for an answer or is over: from the start, from where a
    failed request left it, or on from the answer `answered` names ({"turn", "slot"}). Returns
    whether it is over."""
    config = {"configurable": {"thread_id": thread(interview.id)}}
    async with saver_for(context.session) as saver:
        graph: CompiledStateGraph = build().compile(checkpointer=saver)
        with tracing_context(enabled=False):
            state = await graph.aget_state(config)
            if not state.values:
                start = {"interview_id": interview.id, "size": len(interview.questions), "round": 0}
                await graph.ainvoke(start, config, context=context)
            elif state.next and not state.interrupts:
                await graph.ainvoke(None, config, context=context)
            if answered is not None:
                state = await graph.aget_state(config)
                if state.interrupts:
                    await graph.ainvoke(Command(resume=answered), config, context=context)
            over = not (await graph.aget_state(config)).next
    return over

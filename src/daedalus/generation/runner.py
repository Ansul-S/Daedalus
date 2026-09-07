"""Walking the frozen selection, generating one question per section.

The runner owns three guarantees that the protocol depends on:

**It never generates twice for a section.** A section that already holds a
question, or already holds a rejection, has had its one attempt. Rerunning skips
both. Without that, a rerun would quietly retry every rejected section and turn
a measured rejection rate into a best-of-N result, which is the coaching
`docs/PHASE-6-PROTOCOL.md` section 9 forbids.

**It separates a bad response from an unreachable server.** A response that
fails a deterministic check is a rejection: it is persisted, counted, and never
retried. A transport failure is not — the model was never reached, so the
section has not been generated from, and a later run completes it rather than
regenerating it.

**It does not lose a run to one failure.** Each section is independent; an error
on one is recorded and the walk continues.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import psycopg

from daedalus.generation.client import GenerationError, chat
from daedalus.generation.context import NoProseError, SectionContext, section_context
from daedalus.generation.prompt import (
    GENERATION_MODEL,
    KEEP_ALIVE,
    PROMPT_VERSION,
    RESPONSE_SCHEMA,
    THINK,
    build_messages,
    generation_options,
    params_hash,
)
from daedalus.generation.selection import SelectedSection
from daedalus.generation.validation import Rejection, validate_response
from daedalus.storage.questions import (
    GeneratedQuestion,
    record_question,
    record_rejection,
    rejected_sections,
    sections_with_questions,
)

Connection = psycopg.Connection[tuple[object, ...]]

#: Outcome names used in the report. Stable: they are counted and compared.
ACCEPTED = "accepted"
REJECTED = "rejected"
FAILED = "failed"
SKIPPED = "skipped"


class ChatFunction(Protocol):
    """The shape of the chat call the runner needs.

    Injected rather than imported directly so the runner can be tested against
    recorded responses without a live model.
    """

    def __call__(
        self,
        messages: list[dict[str, str]],
        model: str,
        schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str = KEEP_ALIVE,
        think: bool = THINK,
    ) -> str: ...


@dataclass(frozen=True)
class SectionOutcome:
    """What happened for one section."""

    selection_rank: int
    doc_id: str
    heading_path: tuple[str, ...]
    outcome: str
    reason: str | None = None
    detail: str | None = None
    question_id: int | None = None


@dataclass
class RunReport:
    """Everything one walk of the selection produced."""

    model: str
    prompt_version: str
    params_hash: str
    outcomes: list[SectionOutcome] = field(default_factory=list)

    def _count(self, outcome: str) -> int:
        return sum(1 for item in self.outcomes if item.outcome == outcome)

    @property
    def accepted(self) -> int:
        """Sections that produced a question passing every check."""
        return self._count(ACCEPTED)

    @property
    def rejected(self) -> int:
        """Sections whose response failed a deterministic check."""
        return self._count(REJECTED)

    @property
    def failed(self) -> int:
        """Sections the model could not be reached for."""
        return self._count(FAILED)

    @property
    def skipped(self) -> int:
        """Sections that already had an outcome from an earlier run."""
        return self._count(SKIPPED)

    @property
    def attempted(self) -> int:
        """Sections this run actually called the model for."""
        return self.accepted + self.rejected + self.failed

    def rejection_counts(self) -> dict[str, int]:
        """Rejections by reason, as the protocol requires them reported."""
        counts: dict[str, int] = {}
        for item in self.outcomes:
            if item.outcome == REJECTED and item.reason is not None:
                counts[item.reason] = counts.get(item.reason, 0) + 1
        return counts

    def failure_counts(self) -> dict[str, int]:
        """Transport failures by reason. Not rejections, and reported apart."""
        counts: dict[str, int] = {}
        for item in self.outcomes:
            if item.outcome == FAILED and item.reason is not None:
                counts[item.reason] = counts.get(item.reason, 0) + 1
        return counts


def generate_one(
    selected: SelectedSection,
    context: SectionContext,
    chat_fn: ChatFunction,
    model: str = GENERATION_MODEL,
) -> GeneratedQuestion | Rejection:
    """Generate and check one question. Transport errors propagate.

    A GenerationError is deliberately not caught here. It is not a property of
    the response — there is no response — and collapsing it into a rejection
    would let an unreachable server inflate the rejection rate.
    """
    raw = chat_fn(
        messages=build_messages(context, selected.question_type, selected.difficulty),
        model=model,
        schema=RESPONSE_SCHEMA,
        options=generation_options(selected.section),
        keep_alive=KEEP_ALIVE,
        think=THINK,
    )
    return validate_response(
        raw, selected, context, model, PROMPT_VERSION, params_hash()
    )


def run(
    connection: Connection,
    selection: Sequence[SelectedSection],
    chat_fn: ChatFunction | None = None,
    model: str = GENERATION_MODEL,
    limit: int | None = None,
    on_progress: Callable[[SectionOutcome], None] | None = None,
) -> RunReport:
    """Walk the selection in rank order, generating one question per section.

    Sections are visited in the frozen draw order so that a partial run covers a
    prefix of the selection rather than an arbitrary subset. `limit` caps how
    many sections the model is called for, which is what makes a small
    verification run possible without touching the rest of the draw.
    """
    if limit is not None and limit < 0:
        raise ValueError(f"limit cannot be negative, got {limit}")

    call = chat_fn if chat_fn is not None else _default_chat
    report = RunReport(
        model=model, prompt_version=PROMPT_VERSION, params_hash=params_hash()
    )

    done = sections_with_questions(connection) | rejected_sections(connection)
    attempted = 0

    for selected in sorted(selection, key=lambda item: item.rank):
        section = selected.section
        key = (section.doc_id, section.heading_path)

        if key in done:
            report.outcomes.append(
                _outcome(selected, SKIPPED, "already_attempted", None)
            )
            continue

        if limit is not None and attempted >= limit:
            continue

        attempted += 1
        outcome = _attempt(connection, selected, call, model)
        report.outcomes.append(outcome)
        if on_progress is not None:
            on_progress(outcome)

    return report


def _attempt(
    connection: Connection,
    selected: SelectedSection,
    call: ChatFunction,
    model: str,
) -> SectionOutcome:
    """Generate, check and persist one section, converting failures to outcomes."""
    section = selected.section
    try:
        context = section_context(connection, section.doc_id, section.heading_path)
    except NoProseError as error:
        return _outcome(selected, FAILED, "no_prose", str(error))

    try:
        result = generate_one(selected, context, call, model)
    except GenerationError as error:
        return _outcome(selected, FAILED, "generation_error", str(error))

    if isinstance(result, Rejection):
        record_rejection(
            connection,
            section.doc_id,
            section.heading_path,
            selected.rank,
            result.reason,
            result.detail,
            None,
            model,
            PROMPT_VERSION,
            params_hash(),
        )
        return _outcome(selected, REJECTED, result.reason, result.detail)

    question_id = record_question(connection, result)
    return SectionOutcome(
        selection_rank=selected.rank,
        doc_id=section.doc_id,
        heading_path=section.heading_path,
        outcome=ACCEPTED,
        question_id=question_id,
    )


def _outcome(
    selected: SelectedSection, outcome: str, reason: str | None, detail: str | None
) -> SectionOutcome:
    """Build an outcome for a section that produced no question."""
    return SectionOutcome(
        selection_rank=selected.rank,
        doc_id=selected.section.doc_id,
        heading_path=selected.section.heading_path,
        outcome=outcome,
        reason=reason,
        detail=detail,
    )


def _default_chat(
    messages: list[dict[str, str]],
    model: str,
    schema: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
    keep_alive: str = KEEP_ALIVE,
    think: bool = THINK,
) -> str:
    """Call the real Ollama server. Separated so the default is substitutable."""
    return chat(
        messages=messages,
        model=model,
        schema=schema,
        options=options,
        keep_alive=keep_alive,
        think=think,
    )

"""The deterministic validity checks from `docs/PHASE-6-PROTOCOL.md` section 9.

A response is valid only if all seven checks pass. A failure is recorded with a
reason and the question is discarded: the protocol forbids repairing,
re-prompting or regenerating, because a generator that is retried until it
succeeds has not been measured, it has been coached.

None of these checks involves a judge. They establish that the model returned
what was asked for and was reading the material supplied, which is a
precondition for groundedness rather than evidence of it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace

from daedalus.generation.context import SectionContext
from daedalus.generation.selection import SelectedSection
from daedalus.storage.questions import GeneratedQuestion

#: Rejection reasons. Stable strings: they are reported as counts by reason and
#: must stay comparable across runs.
UNPARSABLE = "unparsable"
SCHEMA_MISMATCH = "schema_mismatch"
EMPTY_QUESTION = "empty_question"
TYPE_MISMATCH = "type_mismatch"
DIFFICULTY_MISMATCH = "difficulty_mismatch"
NO_CITATIONS = "no_citations"
UNKNOWN_CITATION = "unknown_citation"
SEED_NOT_CITED = "seed_not_cited"
EMPTY_QUOTE = "empty_quote"
QUOTE_NOT_FOUND = "quote_not_found"

REJECTION_REASONS = (
    UNPARSABLE,
    SCHEMA_MISMATCH,
    EMPTY_QUESTION,
    TYPE_MISMATCH,
    DIFFICULTY_MISMATCH,
    NO_CITATIONS,
    UNKNOWN_CITATION,
    SEED_NOT_CITED,
    EMPTY_QUOTE,
    QUOTE_NOT_FOUND,
)

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Rejection:
    """A response that failed a check, with why, and the response itself.

    `raw` carries the exact body the model returned. It is the evidence for the
    reason: without it, understanding why a response was rejected means calling
    the model again, which is both slow and — with a fixed decoding seed —
    still not guaranteed to reproduce the same body if anything else changed.
    """

    reason: str
    detail: str
    raw: str = ""


@dataclass(frozen=True)
class ResponseFields:
    """The five fields of a response, once their types have been checked."""

    question: str
    question_type: str
    difficulty: str
    cited_ordinals: tuple[int, ...]
    grounding_quote: str


def normalise(text: str) -> str:
    """Collapse whitespace runs to a single space and strip the ends.

    Applied to both sides of the quote check. A model that reproduces a passage
    correctly but re-wraps it should not be failed for the line breaks, while
    one that paraphrases still is.
    """
    return _WHITESPACE.sub(" ", text).strip()


def _fields(payload: object) -> Rejection | ResponseFields:
    """Return the response fields, or a rejection if the shape is wrong."""
    if not isinstance(payload, dict):
        return Rejection(
            SCHEMA_MISMATCH, f"expected an object, got {type(payload).__name__}"
        )

    for name in ("question", "question_type", "difficulty", "grounding_quote"):
        if name not in payload:
            return Rejection(SCHEMA_MISMATCH, f"missing {name}")
        if not isinstance(payload[name], str):
            return Rejection(SCHEMA_MISMATCH, f"{name} is not a string")

    if "cited_ordinals" not in payload:
        return Rejection(SCHEMA_MISMATCH, "missing cited_ordinals")
    cited = payload["cited_ordinals"]
    if not isinstance(cited, list) or any(
        not isinstance(value, int) or isinstance(value, bool) for value in cited
    ):
        return Rejection(SCHEMA_MISMATCH, "cited_ordinals is not a list of integers")

    return ResponseFields(
        question=payload["question"],
        question_type=payload["question_type"],
        difficulty=payload["difficulty"],
        cited_ordinals=tuple(dict.fromkeys(cited)),
        grounding_quote=payload["grounding_quote"],
    )


def validate_response(
    payload: str,
    selected: SelectedSection,
    context: SectionContext,
    model: str,
    prompt_version: str,
    params_hash: str,
) -> GeneratedQuestion | Rejection:
    """Check one raw response against the protocol, returning it or a rejection.

    Every rejection carries the raw response that produced it, so a stored
    rejection can be understood later without calling the model again.
    """
    result = _check(payload, selected, context, model, prompt_version, params_hash)
    return replace(result, raw=payload) if isinstance(result, Rejection) else result


def _check(
    payload: str,
    selected: SelectedSection,
    context: SectionContext,
    model: str,
    prompt_version: str,
    params_hash: str,
) -> GeneratedQuestion | Rejection:
    """Run the seven checks in the order the protocol lists them.

    The order matters: a response failing several is reported under the first,
    which keeps rejection counts comparable rather than dependent on evaluation
    order.
    """
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as error:
        return Rejection(UNPARSABLE, str(error))

    fields = _fields(parsed)
    if isinstance(fields, Rejection):
        return fields

    if not fields.question.strip():
        return Rejection(EMPTY_QUESTION, "question is blank")

    if fields.question_type != selected.question_type:
        return Rejection(
            TYPE_MISMATCH,
            f"asked for {selected.question_type}, got {fields.question_type!r}",
        )

    if fields.difficulty != selected.difficulty:
        return Rejection(
            DIFFICULTY_MISMATCH,
            f"asked for {selected.difficulty}, got {fields.difficulty!r}",
        )

    cited = fields.cited_ordinals
    if not cited:
        return Rejection(NO_CITATIONS, "cited_ordinals is empty")

    supplied = {chunk.ordinal: chunk for chunk in context.chunks}
    unknown = [ordinal for ordinal in cited if ordinal not in supplied]
    if unknown:
        return Rejection(UNKNOWN_CITATION, f"ordinals not supplied: {sorted(unknown)}")

    if context.seed_ordinal not in cited:
        return Rejection(
            SEED_NOT_CITED, f"seed {context.seed_ordinal} is not among {sorted(cited)}"
        )

    if not fields.grounding_quote.strip():
        return Rejection(EMPTY_QUOTE, "grounding_quote is blank")

    needle = normalise(fields.grounding_quote)
    if not any(needle in normalise(supplied[ordinal].text) for ordinal in cited):
        return Rejection(
            QUOTE_NOT_FOUND,
            f"quote not found verbatim in cited chunks {sorted(cited)}",
        )

    return GeneratedQuestion(
        doc_id=selected.section.doc_id,
        heading_path=selected.section.heading_path,
        seed_ordinal=context.seed_ordinal,
        selection_rank=selected.rank,
        requested_type=selected.question_type,
        requested_difficulty=selected.difficulty,
        text=fields.question.strip(),
        grounding_quote=fields.grounding_quote.strip(),
        context_ordinals=tuple(chunk.ordinal for chunk in context.chunks),
        cited_ordinals=cited,
        model=model,
        prompt_version=prompt_version,
        params_hash=params_hash,
    )

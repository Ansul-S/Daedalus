"""The generation contract: the prompt, the response schema, and the options.

Everything here is fixed by `docs/PHASE-6-PROTOCOL.md` sections 5, 6 and 7 and
was committed before any question was generated. The prompt text is versioned
rather than edited in place: a question generated under one wording is not
comparable to one generated under another, so `PROMPT_VERSION` is stored
alongside every question and changing the text without changing the version
would make the record wrong.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from daedalus.generation.context import SectionContext, render_context
from daedalus.generation.selection import Section, selection_hash

#: Identifies the prompt wording a question was produced under.
PROMPT_VERSION = "p6-v1"

#: Generation model and decoding options. Protocol section 7, itself binding
#: from the measured configuration in docs/PHASE-0.md.
GENERATION_MODEL = "qwen3:8b"
TEMPERATURE = 0.3
NUM_PREDICT = 400
KEEP_ALIVE = "30m"
THINK = False

#: The response the model must return. Protocol section 7.
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "question_type": {
            "type": "string",
            "enum": ["conceptual", "explanation", "comparison", "code_reasoning"],
        },
        "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
        "cited_ordinals": {"type": "array", "items": {"type": "integer"}},
        "grounding_quote": {"type": "string"},
    },
    "required": [
        "question",
        "question_type",
        "difficulty",
        "cited_ordinals",
        "grounding_quote",
    ],
    "additionalProperties": False,
}

#: What each question type is asking for. Stated in the prompt so the model is
#: working to the same definition the interview-relevance rubric will judge it
#: against.
TYPE_INSTRUCTIONS = {
    "conceptual": (
        "Ask what a concept in the material is, or why it holds. The answer "
        "must require understanding the idea, not locating a sentence."
    ),
    "explanation": (
        "Ask the candidate to explain a mechanism or a cause from the material "
        "in their own words."
    ),
    "comparison": (
        "Ask the candidate to compare two things the material actually "
        "discusses, on a dimension the material supports."
    ),
    "code_reasoning": (
        "Ask about what the code in the material does, why it is written that "
        "way, or what would change if it were altered."
    ),
}

#: What each difficulty level means, so the request is a definition rather than
#: a bare word.
DIFFICULTY_INSTRUCTIONS = {
    "easy": ("A candidate who has read this material once should answer it correctly."),
    "medium": (
        "A candidate needs to connect two ideas from the material, or explain "
        "a mechanism rather than name it."
    ),
    "hard": (
        "A candidate needs to reason about consequences, trade-offs or edge "
        "cases that the material supports but does not state outright."
    ),
}

SYSTEM_PROMPT = """\
You write interview questions for an AI/ML interview, grounded strictly in \
material the candidate has studied.

Rules you must follow:

1. Ask exactly one question. Do not answer it.
2. The question must be about the chunk marked (SEED). Other chunks are \
context you may draw on.
3. Ground the question only in the supplied material. Do not use knowledge \
from outside it, and do not invent scenarios, datasets, numbers or systems \
that the material does not mention.
4. A question that can be answered by copying a sentence from the material \
has failed. Require understanding.
5. cited_ordinals must list the ordinals of the chunks the question relies \
on. It must include the SEED chunk's ordinal, and every ordinal must be one \
that appears in the supplied material.
6. grounding_quote must be copied word for word from one of the chunks you \
cited. Do not paraphrase it, do not shorten it with ellipses, and do not \
quote a chunk you did not cite.
7. Return the requested question_type and difficulty exactly as asked.
"""


def build_messages(
    context: SectionContext, question_type: str, difficulty: str
) -> list[dict[str, str]]:
    """Build the chat messages for one section.

    The requested type and difficulty are stated in the instruction rather than
    left for the model to infer, because both are assigned deterministically by
    the protocol and echoed back for checking.
    """
    if question_type not in TYPE_INSTRUCTIONS:
        raise ValueError(f"unknown question type {question_type!r}")
    if difficulty not in DIFFICULTY_INSTRUCTIONS:
        raise ValueError(f"unknown difficulty {difficulty!r}")

    user = f"""\
MATERIAL

{render_context(context)}
REQUEST

question_type: {question_type}
{TYPE_INSTRUCTIONS[question_type]}

difficulty: {difficulty}
{DIFFICULTY_INSTRUCTIONS[difficulty]}

The SEED chunk is [{context.doc_id}:{context.seed_ordinal}]. Your question \
must be about it, and {context.seed_ordinal} must appear in cited_ordinals.
"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def generation_seed(section: Section) -> int:
    """Return the decoding seed for a section.

    The first 8 hex digits of the section's selection hash, as an integer, so
    that a rerun reproduces the same output and the seed is not a free
    parameter anyone could tune.
    """
    return int(selection_hash(section)[:8], 16)


def generation_options(section: Section) -> dict[str, Any]:
    """Return the Ollama decoding options for one section."""
    return {
        "temperature": TEMPERATURE,
        "num_predict": NUM_PREDICT,
        "seed": generation_seed(section),
    }


def params_hash() -> str:
    """Return a hash of everything that fixes how generation behaves.

    Stored with each question so that output produced under different settings
    cannot be silently pooled with output produced under these.
    """
    payload = json.dumps(
        {
            "model": GENERATION_MODEL,
            "prompt_version": PROMPT_VERSION,
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT,
            "think": THINK,
            "keep_alive": KEEP_ALIVE,
            "system_prompt": SYSTEM_PROMPT,
            "schema": RESPONSE_SCHEMA,
        },
        sort_keys=True,
    )
    return hashlib.md5(payload.encode()).hexdigest()

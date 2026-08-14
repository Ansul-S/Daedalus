"""
The prompts that ground an answer in retrieved material.

Prompts live here rather than in ``llm/`` on a deliberate rule: ``llm/``
is *how to talk to a model* and is task-agnostic, while ``generation/`` is
*what to ask and what to do with the reply*. A prompt is a task
definition, so it belongs on this side of the line. That is also what
keeps a single `prompts` concept from existing in two places.

The system prompt carries the only behaviour that matters for evaluation:
answer from the sources or say you cannot. A model that answers plausibly
from its own pretraining is the failure mode the unanswerable slice of the
benchmark exists to catch, and no amount of retrieval quality fixes it.
"""

from __future__ import annotations

from collections.abc import Sequence

from daedalus.storage.types import ChunkRecord

__all__ = ["REFUSAL", "SYSTEM_PROMPT", "build_prompt", "format_sources"]


# Returned without calling the model when retrieval found nothing. Fixed
# text rather than a generated apology: it costs nothing, it cannot
# hallucinate, and the evaluation harness can match on it exactly.
REFUSAL = "The study material provided does not cover this question."


SYSTEM_PROMPT = f"""\
You are a study assistant for AI and machine learning students. You answer \
strictly from the numbered sources you are given.

Rules:
- Use only the sources. Do not add facts from your own knowledge, even if \
you are confident they are correct.
- Cite every claim with the source number in square brackets, like [1]. A \
sentence drawn from more than one source cites each, like [1][3].
- If the sources do not answer the question, reply with exactly: {REFUSAL}
- Do not speculate, and do not describe what the sources fail to say beyond \
that one sentence.
- Be concise and concrete. Prefer the source's own terminology.\
"""


def format_sources(records: Sequence[ChunkRecord]) -> str:
    """
    Render retrieved chunks as a numbered list the model can cite.

    Numbering is one-based and positional: source ``[1]`` is the first
    record given. The citation parser resolves markers back through the
    same ordering, so the two must not drift apart.
    """

    blocks = []

    for index, record in enumerate(records, start=1):
        # The origin is included because a student needs to know which
        # document to go read, and the page is what makes that actionable.
        location = f"{record.doc_id}"

        if record.page is not None:
            location += f", page {record.page}"

        blocks.append(f"[{index}] ({location})\n{record.text}")

    return "\n\n".join(blocks)


def build_prompt(question: str, records: Sequence[ChunkRecord]) -> str:
    """Assemble the user prompt: the sources, then the question."""

    # Question last. Instructions at the end of a long context are followed
    # more reliably than instructions buried before it.
    return f"Sources:\n\n{format_sources(records)}\n\nQuestion: {question}"

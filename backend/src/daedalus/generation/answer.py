"""
Grounded answering: retrieve, generate, attribute.

The last step of the vertical slice, and the one that decides whether any
of the rest was worth building. Retrieval can be perfect and the answer
still ungrounded, so what comes back is not just text — it is text plus
the sources the model actually cited, which is what makes the claim
checkable by a reader and scoreable by the evaluation harness.

Two behaviours are worth stating outright:

**An empty retrieval never reaches the model.** With no sources there is
nothing to ground an answer in, and asking anyway invites exactly the
confident fabrication the benchmark's unanswerable slice measures.

**A refusal is not detected here.** Whether prose counts as a refusal is a
judgement, and judgement belongs in the evaluation harness with a rubric,
not in a regex in the serving path.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

from daedalus.config import constants
from daedalus.generation.prompts import REFUSAL, SYSTEM_PROMPT, build_prompt
from daedalus.interfaces.embedding import Embedder
from daedalus.interfaces.llm import LLM
from daedalus.retrieval import HybridRetriever
from daedalus.storage import chunks as chunk_store
from daedalus.storage.types import ChunkRecord

__all__ = ["Answer", "Source", "answer_question", "cited_indices"]


logger = logging.getLogger(__name__)


_CITATION = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class Source:
    """One numbered source offered to the model."""

    index: int
    chunk_id: int
    doc_id: str
    ordinal: int
    text: str
    source_start: int
    source_end: int
    extraction: str
    page: int | None = None

    @classmethod
    def from_record(cls, index: int, record: ChunkRecord) -> Source:
        return cls(
            index=index,
            chunk_id=record.id,
            doc_id=record.doc_id,
            ordinal=record.ordinal,
            text=record.text,
            source_start=record.source_start,
            source_end=record.source_end,
            extraction=record.extraction,
            page=record.page,
        )


@dataclass(frozen=True)
class Answer:
    """
    A generated answer and its provenance.

    ``sources`` is everything retrieved and shown to the model;
    ``citations`` is the subset it actually referenced. Keeping both is
    what lets the harness separate a retrieval failure from a grounding
    failure — the right chunk being absent is a different bug from the
    right chunk being present and ignored.
    """

    question: str
    text: str
    citations: tuple[Source, ...]
    sources: tuple[Source, ...]
    model: str


def cited_indices(text: str, available: int) -> list[int]:
    """
    Extract the source numbers an answer refers to, in order of first use.

    Markers outside the range offered are dropped rather than resolved: a
    model that invents ``[7]`` when it was given four sources must not be
    able to index past the end of the list, and a fabricated citation is
    not evidence of anything.
    """

    seen: list[int] = []

    for match in _CITATION.finditer(text):
        index = int(match.group(1))

        if 1 <= index <= available and index not in seen:
            seen.append(index)

    return seen


def answer_question(
    connection: sqlite3.Connection,
    embedder: Embedder,
    llm: LLM,
    question: str,
    *,
    top_k: int = constants.DEFAULT_TOP_K,
) -> Answer:
    """Retrieve context for a question and answer from it."""

    hits = HybridRetriever(connection, embedder).search(question, top_k)
    records: Sequence[ChunkRecord] = chunk_store.fetch(connection, [hit.chunk_id for hit in hits])

    sources = tuple(
        Source.from_record(index, record) for index, record in enumerate(records, start=1)
    )

    if not sources:
        logger.info("No sources for %r; refusing without calling the model", question)

        return Answer(
            question=question,
            text=REFUSAL,
            citations=(),
            sources=(),
            model=llm.model,
        )

    text = llm.complete(build_prompt(question, records), system=SYSTEM_PROMPT)

    citations = tuple(sources[index - 1] for index in cited_indices(text, len(sources)))

    logger.info("Answered %r from %d sources, %d cited", question, len(sources), len(citations))

    return Answer(
        question=question,
        text=text,
        citations=citations,
        sources=sources,
        model=llm.model,
    )

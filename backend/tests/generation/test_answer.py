"""
Grounded answering.

The model is scripted, so what is under test is the wiring around it: that
retrieved context reaches the prompt, that citation markers resolve to the
right chunks, and that a question with no sources never reaches the model
at all.
"""

from __future__ import annotations

import sqlite3

import pytest

from daedalus.embeddings import FakeEmbedder
from daedalus.generation import answer_question
from daedalus.generation.answer import cited_indices
from daedalus.generation.prompts import REFUSAL
from daedalus.ingestion.types import Chunk
from daedalus.interfaces.embedding import Embedder
from daedalus.llm import FakeLLM
from daedalus.storage import chunks as chunk_store
from daedalus.storage import documents

CORPUS = [
    "the dot product is scaled by the square root of d_k before the softmax",
    "layer normalization stabilizes training by rescaling activations",
    "convolution kernels slide across an image to extract local features",
]


@pytest.fixture
def embedder() -> Embedder:
    return FakeEmbedder()


@pytest.fixture
def indexed(db: sqlite3.Connection, embedder: Embedder) -> sqlite3.Connection:
    documents.create(
        db,
        doc_id="attention",
        filename="attention.pdf",
        source_type="arxiv",
        content_hash="hash-attention",
    )

    chunks = [
        Chunk(
            ordinal=ordinal,
            text=text,
            source_start=ordinal * 1000,
            source_end=ordinal * 1000 + len(text),
            extraction="text",
            page=ordinal + 1,
        )
        for ordinal, text in enumerate(CORPUS)
    ]

    chunk_store.replace(db, "attention", chunks, embedder.embed_documents(CORPUS))

    return db


# Citation Parsing


def test_a_marker_resolves_to_its_source() -> None:
    assert cited_indices("Because of scaling [2].", available=3) == [2]


def test_several_markers_keep_first_use_order() -> None:
    assert cited_indices("First [3], then [1], again [3].", available=3) == [3, 1]


def test_an_invented_citation_is_dropped() -> None:
    """A model given four sources must not be able to index past them."""

    assert cited_indices("As shown [7].", available=4) == []


def test_a_zero_citation_is_dropped() -> None:
    assert cited_indices("See [0].", available=3) == []


def test_text_without_citations_has_none() -> None:
    assert cited_indices("A confident, unsourced claim.", available=3) == []


# Answering


def test_the_answer_is_returned(indexed: sqlite3.Connection, embedder: Embedder) -> None:
    result = answer_question(indexed, embedder, FakeLLM(default="Scaled. [1]"), "softmax")

    assert result.text == "Scaled. [1]"


def test_the_retrieved_context_reaches_the_prompt(
    indexed: sqlite3.Connection, embedder: Embedder
) -> None:
    """Grounding is impossible if the sources never arrive."""

    llm = FakeLLM()

    answer_question(indexed, embedder, llm, "softmax")

    assert "square root of d_k" in llm.prompts[0]


def test_the_question_reaches_the_prompt(indexed: sqlite3.Connection, embedder: Embedder) -> None:
    llm = FakeLLM()

    answer_question(indexed, embedder, llm, "why scale the dot product")

    assert "why scale the dot product" in llm.prompts[0]


def test_the_grounding_rules_are_sent_as_a_system_prompt(
    indexed: sqlite3.Connection, embedder: Embedder
) -> None:
    llm = FakeLLM()

    answer_question(indexed, embedder, llm, "softmax")

    assert llm.systems[0] is not None
    assert "only the sources" in llm.systems[0]


def test_a_citation_resolves_to_the_chunk_it_names(
    indexed: sqlite3.Connection, embedder: Embedder
) -> None:
    llm = FakeLLM(default="See the first source. [1]")

    result = answer_question(indexed, embedder, llm, "softmax", top_k=3)

    assert len(result.citations) == 1
    assert result.citations[0].index == 1
    assert result.citations[0].chunk_id == result.sources[0].chunk_id


def test_citations_carry_the_offsets_a_reader_needs(
    indexed: sqlite3.Connection, embedder: Embedder
) -> None:
    result = answer_question(indexed, embedder, FakeLLM(), "softmax")

    citation = result.citations[0]

    assert citation.doc_id == "attention"
    assert citation.source_end > citation.source_start
    assert citation.page is not None


def test_an_uncited_answer_has_no_citations(
    indexed: sqlite3.Connection, embedder: Embedder
) -> None:
    """Everything retrieved is still reported as a source."""

    llm = FakeLLM(default="An unsourced claim.")

    result = answer_question(indexed, embedder, llm, "softmax", top_k=3)

    assert result.citations == ()
    assert result.sources


def test_sources_are_numbered_from_one(indexed: sqlite3.Connection, embedder: Embedder) -> None:
    result = answer_question(indexed, embedder, FakeLLM(), "softmax", top_k=3)

    assert [source.index for source in result.sources] == [1, 2, 3]


def test_top_k_limits_the_sources(indexed: sqlite3.Connection, embedder: Embedder) -> None:
    result = answer_question(indexed, embedder, FakeLLM(), "softmax", top_k=2)

    assert len(result.sources) == 2


def test_the_model_is_recorded(indexed: sqlite3.Connection, embedder: Embedder) -> None:
    """A metric is meaningless without knowing what produced it."""

    result = answer_question(indexed, embedder, FakeLLM(), "softmax")

    assert result.model == "fake"


# Refusal


def test_an_empty_index_refuses(db: sqlite3.Connection, embedder: Embedder) -> None:
    result = answer_question(db, embedder, FakeLLM(), "anything")

    assert result.text == REFUSAL
    assert result.citations == ()
    assert result.sources == ()


def test_a_refusal_never_calls_the_model(db: sqlite3.Connection, embedder: Embedder) -> None:
    """With no sources there is nothing to ground an answer in."""

    llm = FakeLLM()

    answer_question(db, embedder, llm, "anything")

    assert llm.prompts == []

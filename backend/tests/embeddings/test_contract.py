"""
The contract every Embedder must satisfy.

Written once and run against each implementation, so the fake cannot drift
away from the real model in any way retrieval depends on. If a future
adapter passes these, it is safe to substitute.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from daedalus.config import constants
from daedalus.embeddings import BGEEmbedder, FakeEmbedder
from daedalus.interfaces import Embedder

TEXTS = [
    "Attention is all you need.",
    "Retrieval-augmented generation grounds answers in sources.",
    "The mitochondrion is the powerhouse of the cell.",
]


def _real_embedder() -> Embedder:
    """
    Build the real embedder, or skip.

    Gated on an environment variable and not merely on the import, so that a
    developer who has synced the ml extra still gets a fast test run — the
    first call downloads roughly 2.2 GB of weights.
    """

    if os.environ.get("DAEDALUS_TEST_REAL_EMBEDDER") != "1":
        pytest.skip("set DAEDALUS_TEST_REAL_EMBEDDER=1 to test against the real model")

    pytest.importorskip("sentence_transformers")

    return BGEEmbedder()


@pytest.fixture(params=["fake", "real"])
def embedder(request: pytest.FixtureRequest) -> Embedder:
    if request.param == "fake":
        return FakeEmbedder()

    return _real_embedder()


def test_reports_the_dimension_the_index_expects(embedder: Embedder) -> None:
    assert embedder.dim == constants.EMBEDDING_DIM


def test_documents_produce_one_row_each(embedder: Embedder) -> None:
    matrix = embedder.embed_documents(TEXTS)

    assert matrix.shape == (len(TEXTS), embedder.dim)


def test_vectors_are_float32(embedder: Embedder) -> None:
    matrix = embedder.embed_documents(TEXTS)

    assert matrix.dtype == np.float32


def test_rows_are_unit_length(embedder: Embedder) -> None:
    matrix = embedder.embed_documents(TEXTS)
    norms = np.linalg.norm(matrix, axis=1)

    assert np.allclose(norms, 1.0, atol=1e-5)


def test_rows_follow_input_order(embedder: Embedder) -> None:
    """Row i must be the embedding of texts[i], not of some other input."""

    matrix = embedder.embed_documents(TEXTS)

    for row, text in enumerate(TEXTS):
        alone = embedder.embed_documents([text])[0]

        assert np.allclose(matrix[row], alone, atol=1e-4)


def test_empty_input_returns_an_empty_matrix(embedder: Embedder) -> None:
    matrix = embedder.embed_documents([])

    assert matrix.shape == (0, embedder.dim)


def test_query_returns_a_single_vector(embedder: Embedder) -> None:
    vector = embedder.embed_query("What is attention?")

    assert vector.shape == (embedder.dim,)
    assert vector.dtype == np.float32
    assert np.isclose(float(np.linalg.norm(vector)), 1.0, atol=1e-5)


def test_encoding_is_deterministic(embedder: Embedder) -> None:
    """A stored chunk must still match itself on the next run."""

    first = embedder.embed_documents(TEXTS)
    second = embedder.embed_documents(TEXTS)

    assert np.allclose(first, second, atol=1e-6)


def test_query_and_document_agree_on_identical_text(embedder: Embedder) -> None:
    """
    A query identical to a chunk must be its nearest neighbour.

    This is the property that would break silently if an asymmetric model
    were dropped in without its prefixes.
    """

    text = TEXTS[0]

    assert np.allclose(embedder.embed_query(text), embedder.embed_documents([text])[0], atol=1e-4)


def test_embedder_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        Embedder()  # type: ignore[abstract]

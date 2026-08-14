"""
Properties specific to the fake embedder.

The shared contract test covers what every implementation owes its callers.
These cover what makes the fake usable as a test double — reproducibility
that survives a new process, and vectors a test can pin.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import daedalus
from daedalus.config import constants
from daedalus.embeddings import FakeEmbedder


def test_same_text_gives_the_same_vector() -> None:
    embedder = FakeEmbedder()

    assert np.array_equal(embedder.embed_query("attention"), embedder.embed_query("attention"))


def test_separate_instances_agree() -> None:
    assert np.array_equal(
        FakeEmbedder().embed_query("attention"),
        FakeEmbedder().embed_query("attention"),
    )


def test_different_texts_give_different_vectors() -> None:
    embedder = FakeEmbedder()

    assert not np.allclose(embedder.embed_query("attention"), embedder.embed_query("retrieval"))


def test_vectors_survive_a_new_process() -> None:
    """
    Seeded from SHA-256, not Python's salted ``hash()``.

    Regression guard: with ``hash()``, vectors written to the database in
    one run would fail to match the same text in the next, and the bug would
    look like a retrieval-quality problem rather than a seeding problem.
    """

    script = (
        "from daedalus.embeddings import FakeEmbedder;"
        "print(float(FakeEmbedder().embed_query('attention')[0]))"
    )
    # pytest's pythonpath setting applies to this process only, so the child
    # is told where the package lives explicitly.
    src_dir = Path(daedalus.__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(src_dir)},
    )

    in_process = float(FakeEmbedder().embed_query("attention")[0])

    assert float(result.stdout.strip()) == pytest.approx(in_process)


def test_dimension_is_configurable() -> None:
    embedder = FakeEmbedder(dim=8)

    assert embedder.dim == 8
    assert embedder.embed_documents(["a", "b"]).shape == (2, 8)


def test_default_dimension_matches_the_index() -> None:
    assert FakeEmbedder().dim == constants.EMBEDDING_DIM


def test_rejects_a_nonsense_dimension() -> None:
    with pytest.raises(ValueError, match="positive"):
        FakeEmbedder(dim=0)


def test_overrides_pin_specific_vectors() -> None:
    """Lets a retrieval test lay out a known neighbourhood."""

    embedder = FakeEmbedder(dim=4, overrides={"north": [1.0, 0.0, 0.0, 0.0]})

    assert np.allclose(embedder.embed_query("north"), [1.0, 0.0, 0.0, 0.0])


def test_overrides_are_normalized() -> None:
    embedder = FakeEmbedder(dim=4, overrides={"long": [3.0, 4.0, 0.0, 0.0]})

    assert np.allclose(embedder.embed_query("long"), [0.6, 0.8, 0.0, 0.0])


def test_overrides_must_match_the_dimension() -> None:
    with pytest.raises(ValueError, match="expected 4"):
        FakeEmbedder(dim=4, overrides={"short": [1.0, 0.0]})


def test_unpinned_text_still_gets_a_hashed_vector() -> None:
    embedder = FakeEmbedder(dim=4, overrides={"north": [1.0, 0.0, 0.0, 0.0]})
    other = embedder.embed_query("south")

    assert float(np.linalg.norm(other)) == pytest.approx(1.0, abs=1e-5)
    assert not np.allclose(other, [1.0, 0.0, 0.0, 0.0])

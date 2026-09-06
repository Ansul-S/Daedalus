"""Tests for the RRF fusion used by the hybrid experiment.

The fusion lives in `scripts/evaluate_hybrid.py`, which is experiment code
rather than part of the package, so it is loaded by path. It is tested anyway: a
fusion bug would produce a wrong benchmark number silently, which is the failure
mode the protocol exists to prevent.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_hybrid.py"

# Compiled from source text rather than loaded through SourceFileLoader, which
# consults __pycache__. A stale .pyc there once served an older version of this
# script to the tests, so a mutation could pass while the change was live on
# disk. Reading the file directly cannot go stale.
hybrid = ModuleType("evaluate_hybrid")
hybrid.__file__ = str(_PATH)
exec(compile(_PATH.read_text(), str(_PATH), "exec"), hybrid.__dict__)

rrf = hybrid.reciprocal_rank_fusion


def test_the_frozen_constants_are_what_the_protocol_declares() -> None:
    """A guard on the pre-registration, not a tautology.

    These values are fixed in docs/HYBRID-PROTOCOL.md before the run. If one
    changes, this test fails and the change has to be deliberate.
    """
    assert hybrid.RRF_K == 60
    assert hybrid.DEPTH == 10
    assert hybrid.KS == (1, 3, 5, 7, 10)
    assert hybrid.THRESHOLDS == (1, 2)
    assert hybrid.PRIMARY_METRIC == "ndcg@10"
    assert hybrid.PRIMARY_BASELINE == "bge-m3 (production)"


def test_a_single_list_is_returned_in_its_own_order() -> None:
    ranking = [("d", 1), ("d", 2), ("d", 3)]
    assert rrf([ranking]) == ranking


def test_each_chunk_appears_exactly_once() -> None:
    a = [("d", 1), ("d", 2), ("d", 3)]
    b = [("d", 3), ("d", 2), ("d", 1)]
    fused = rrf([a, b])
    assert len(fused) == len(set(fused)) == 3


def test_agreement_beats_a_high_rank_in_one_list() -> None:
    """At k=60 and depth 10, 2/70 > 1/61, so consensus wins outright.

    The protocol records this consequence in advance rather than treating it as
    a surprise.
    """
    a = [("d", 99)] + [("d", i) for i in range(1, 10)] + [("d", 50)]
    b = [("d", 98)] + [("d", i) for i in range(20, 29)] + [("d", 50)]
    fused = rrf([a, b])
    assert fused[0] == ("d", 50)  # rank 10 in both
    assert fused.index(("d", 50)) < fused.index(("d", 99))  # beats rank 1


def test_score_is_the_sum_of_reciprocal_ranks() -> None:
    """Ordering must follow 1/(k+rank) summed, checked on a hand-worked case."""
    a = [("d", 1), ("d", 2)]
    b = [("d", 2), ("d", 1)]
    # both chunks: 1/61 + 1/62, identical -- so the tie-break decides
    assert rrf([a, b]) == [("d", 1), ("d", 2)]

    c = [("d", 1), ("d", 2)]
    d = [("d", 1), ("d", 2)]
    # ("d",1) scores 2/61, ("d",2) scores 2/62
    assert rrf([c, d]) == [("d", 1), ("d", 2)]


def test_ties_are_broken_on_doc_id_then_ordinal() -> None:
    a = [("b", 5), ("a", 9)]
    b = [("a", 9), ("b", 5)]
    assert rrf([a, b]) == [("a", 9), ("b", 5)]

    c = [("a", 2), ("a", 1)]
    d = [("a", 1), ("a", 2)]
    assert rrf([c, d]) == [("a", 1), ("a", 2)]


def test_fusion_is_deterministic_across_repeats() -> None:
    a = [("d", i) for i in range(10)]
    b = [("d", i) for i in range(5, 15)]
    first = rrf([a, b])
    for _ in range(5):
        assert rrf([a, b]) == first


def test_input_order_of_the_lists_does_not_change_the_result() -> None:
    """Unweighted fusion: neither retriever is privileged by argument position."""
    a = [("d", i) for i in range(10)]
    b = [("d", i) for i in range(5, 15)]
    assert rrf([a, b]) == rrf([b, a])


def test_empty_inputs() -> None:
    assert rrf([]) == []
    assert rrf([[], []]) == []
    assert rrf([[("d", 1)], []]) == [("d", 1)]


def test_k_changes_the_ordering_as_expected() -> None:
    """A guard that k is genuinely used, not ignored.

    With a small k, being rank 1 in one list can beat being low in both. That is
    exactly why k is frozen at the published 60 rather than chosen here.
    """
    a = [("d", 1)] + [("d", i) for i in range(10, 19)] + [("d", 2)]
    b = [("d", 3)] + [("d", i) for i in range(20, 29)] + [("d", 2)]
    assert rrf([a, b], k=60)[0] == ("d", 2)
    assert rrf([a, b], k=1)[0] == ("d", 1)


def test_fusion_output_is_accepted_by_the_metrics() -> None:
    """metrics.py rejects a repeated reference, so this proves de-duplication."""
    from daedalus.evaluation.metrics import precision_at_k

    a = [("d", 1), ("d", 2), ("d", 3)]
    b = [("d", 3), ("d", 1), ("d", 4)]
    fused = rrf([a, b])
    grades = {("d", 1): 2, ("d", 2): 0, ("d", 3): 1, ("d", 4): 0}
    assert precision_at_k(fused, grades, 4, 1, "zero") == pytest.approx(0.5)


def test_ranks_are_counted_from_one_not_zero() -> None:
    """Pins the 1-indexed rank of the published RRF definition.

    Every other test here passes under a 0-indexed loop: the consensus-beats-
    singleton property and within-list ordering both survive an off-by-one, and
    only the scores shift. This case is built so the error changes the ORDER.

    With k=1 and 1-indexed ranks, ("b",1) at rank 1 in one list scores 1/2, and
    ("a",1) at rank 3 in both scores 1/4 + 1/4 = 1/2. They tie, so the tie-break
    puts ("a",1) first. Counting from zero makes them 1/1 and 2/3, and ("b",1)
    wins outright.
    """
    first = [("b", 1), ("z", 1), ("a", 1)]
    second = [("y", 1), ("x", 1), ("a", 1)]
    assert rrf([first, second], k=1)[0] == ("a", 1)

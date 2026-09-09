"""Tests for judge-versus-human agreement.

Every expected value here is worked out by hand in the test that uses it, so a
change in the implementation cannot quietly redefine what the numbers mean.
"""

from __future__ import annotations

import pytest

from daedalus.evaluation import agreement

# Ten pairs with six matches, used for both kappa tests below.
#
#   human   0 0 0 0 0   1 1 1   2 2
#   judge   0 0 0 1 1   1 1 2   2 0
#
# raw agreement 6/10; human marginals 0:5 1:3 2:2; judge marginals 0:4 1:4 2:2.
WORKED = [
    ("0", "0"),
    ("0", "0"),
    ("0", "0"),
    ("0", "1"),
    ("0", "1"),
    ("1", "1"),
    ("1", "1"),
    ("1", "2"),
    ("2", "2"),
    ("2", "0"),
]


def test_raw_agreement_counts_exact_matches() -> None:
    assert agreement.raw_agreement(WORKED) == pytest.approx(0.6)


def test_raw_agreement_is_undefined_on_no_pairs() -> None:
    assert agreement.raw_agreement([]) is None


def test_unweighted_kappa_matches_a_hand_calculation() -> None:
    # Po = 0.6. Pe = (5*4 + 3*4 + 2*2) / 100 = 0.36.
    # kappa = (0.6 - 0.36) / (1 - 0.36) = 0.375.
    kappa = agreement.cohen_kappa(WORKED, ("0", "1", "2"), "none")
    assert kappa == pytest.approx(0.375)


def test_linear_weights_credit_a_one_step_disagreement() -> None:
    # Observed weighted disagreement = (2*0.5 + 0.5 + 1.0) / 10 = 0.25.
    # Expected = 41 / 100 = 0.41.  kappa_w = 1 - 0.25 / 0.41.
    kappa = agreement.cohen_kappa(WORKED, ("0", "1", "2"), "linear")
    assert kappa == pytest.approx(1.0 - 0.25 / 0.41)


def test_weighting_credits_the_near_misses_unweighted_kappa_discards() -> None:
    unweighted = agreement.cohen_kappa(WORKED, ("0", "1", "2"), "none")
    weighted = agreement.cohen_kappa(WORKED, ("0", "1", "2"), "linear")

    assert unweighted is not None and weighted is not None
    assert weighted > unweighted


def test_one_step_errors_score_higher_than_two_step_errors() -> None:
    """Two datasets, three matches each; only the size of the misses differs."""
    one_step = [("0", "0"), ("1", "1"), ("2", "2"), ("0", "1"), ("1", "0")]
    two_step = [("0", "0"), ("1", "1"), ("2", "2"), ("0", "2"), ("2", "0")]

    near = agreement.cohen_kappa(one_step, ("0", "1", "2"), "linear")
    far = agreement.cohen_kappa(two_step, ("0", "1", "2"), "linear")

    assert near == pytest.approx(0.5)
    assert far == pytest.approx(1.0 - 0.4 / 0.48)
    assert near is not None and far is not None and near > far


def test_perfect_agreement_is_one() -> None:
    pairs = [("0", "0"), ("1", "1"), ("2", "2")]

    assert agreement.raw_agreement(pairs) == pytest.approx(1.0)
    assert agreement.cohen_kappa(pairs, ("0", "1", "2"), "none") == pytest.approx(1.0)
    assert agreement.cohen_kappa(pairs, ("0", "1", "2"), "linear") == pytest.approx(1.0)


def test_kappa_is_undefined_when_both_raters_use_one_category() -> None:
    """No expected disagreement means no chance-corrected question to answer."""
    pairs = [("2", "2")] * 8

    assert agreement.cohen_kappa(pairs, ("0", "1", "2"), "none") is None


def test_an_unused_category_does_not_move_the_result() -> None:
    """It contributes nothing to expected disagreement, and the weights cancel."""
    pairs = [("0", "0"), ("0", "1"), ("1", "1"), ("1", "0")]

    two = agreement.cohen_kappa(pairs, ("0", "1"), "linear")
    three = agreement.cohen_kappa(pairs, ("0", "1", "2"), "linear")

    assert two == pytest.approx(three)


def test_the_declared_order_decides_which_labels_are_adjacent() -> None:
    """Reordering the scale changes what counts as a one-step disagreement."""
    ordered = agreement.cohen_kappa(WORKED, ("0", "1", "2"), "linear")
    shuffled = agreement.cohen_kappa(WORKED, ("1", "0", "2"), "linear")

    assert ordered is not None and shuffled is not None
    assert ordered != pytest.approx(shuffled)


def test_kappa_rejects_labels_outside_the_declared_scale() -> None:
    with pytest.raises(ValueError, match="outside the declared scale"):
        agreement.cohen_kappa([("0", "maybe")], ("0", "1", "2"), "none")


def test_kappa_rejects_unknown_weights() -> None:
    with pytest.raises(ValueError, match="unknown weights"):
        agreement.cohen_kappa(WORKED, ("0", "1", "2"), "quadratic")  # type: ignore[arg-type]


def test_categories_add_unusable_only_for_difficulty() -> None:
    assert agreement.categories("difficulty", "include") == (
        "easy",
        "medium",
        "hard",
        "unusable",
    )
    assert agreement.categories("difficulty", "exclude") == ("easy", "medium", "hard")
    assert agreement.categories("relevance", "include") == ("0", "1", "2")


def test_a_set_containing_unusable_is_not_ordinal() -> None:
    assert agreement.is_ordinal("difficulty", "exclude") is True
    assert agreement.is_ordinal("difficulty", "include") is False
    assert agreement.is_ordinal("groundedness", "include") is True


def test_excluding_unusable_drops_the_pair_and_counts_it() -> None:
    pairs = [("easy", "easy"), ("unusable", "hard"), ("hard", "hard")]

    kept, dropped, off = agreement.prepare(pairs, "difficulty", "exclude", "exclude")

    assert kept == [("easy", "easy"), ("hard", "hard")]
    assert dropped == 1
    assert off == 0


def test_including_unusable_keeps_it_as_a_category() -> None:
    pairs = [("unusable", "unusable"), ("easy", "easy")]

    kept, dropped, off = agreement.prepare(pairs, "difficulty", "include", "exclude")

    assert kept == pairs
    assert dropped == 0


def test_unusable_is_off_scale_for_the_judge_when_excluded() -> None:
    """With unusable excluded, a judge that answers unusable is off the scale."""
    pairs = [("easy", "unusable"), ("hard", "hard")]

    kept, _, off = agreement.prepare(pairs, "difficulty", "exclude", "exclude")

    assert kept == [("hard", "hard")]
    assert off == 1


def test_off_scale_can_be_kept_as_a_disagreement() -> None:
    pairs = [("0", "banana"), ("1", "1")]

    kept, _, off = agreement.prepare(pairs, "groundedness", "exclude", "disagree")

    assert kept == pairs
    assert off == 1
    assert agreement.raw_agreement(kept) == pytest.approx(0.5)


def test_a_human_label_off_the_scale_is_a_defect_not_a_policy() -> None:
    with pytest.raises(ValueError, match="not on the groundedness scale"):
        agreement.prepare([("3", "0")], "groundedness", "exclude", "exclude")


def test_prepare_rejects_unknown_rubrics_and_policies() -> None:
    with pytest.raises(ValueError, match="unknown rubric"):
        agreement.prepare([], "clarity", "exclude", "exclude")
    with pytest.raises(ValueError, match="unknown unusable policy"):
        agreement.prepare([], "relevance", "keep", "exclude")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown off-scale policy"):
        agreement.prepare([], "relevance", "exclude", "ignore")  # type: ignore[arg-type]


def test_measure_reports_what_the_denominator_cost() -> None:
    pairs = [
        ("easy", "easy"),
        ("medium", "hard"),
        ("unusable", "easy"),
        ("hard", "banana"),
    ]

    result = agreement.measure(pairs, "difficulty", "exclude", "exclude")

    assert result.n == 2
    assert result.unusable_dropped == 1
    assert result.off_scale == 1
    assert result.weights == "linear"
    assert result.human_marginals == {"easy": 1, "medium": 1}


def test_measure_uses_no_weights_once_unusable_is_a_category() -> None:
    pairs = [("easy", "easy"), ("unusable", "unusable"), ("hard", "medium")]

    result = agreement.measure(pairs, "difficulty", "include", "exclude")

    assert result.weights == "none"
    assert result.n == 3


def test_measure_withholds_kappa_when_off_scale_labels_are_kept() -> None:
    """A chance-corrected rate needs an expected rate for every category."""
    pairs = [("0", "0"), ("1", "banana"), ("2", "2")]

    kept = agreement.measure(pairs, "groundedness", "exclude", "disagree")
    dropped = agreement.measure(pairs, "groundedness", "exclude", "exclude")

    assert kept.kappa is None
    assert kept.raw == pytest.approx(2 / 3)
    assert dropped.kappa is not None
    assert dropped.n == 2


def test_measure_still_reports_kappa_when_nothing_was_off_scale() -> None:
    pairs = [("0", "0"), ("1", "1"), ("2", "0")]

    result = agreement.measure(pairs, "groundedness", "exclude", "disagree")

    assert result.off_scale == 0
    assert result.kappa is not None


def test_the_bootstrap_is_reproducible_from_its_seed() -> None:
    first = agreement.bootstrap_ci(WORKED, agreement.raw_agreement, resamples=200)
    again = agreement.bootstrap_ci(WORKED, agreement.raw_agreement, resamples=200)

    assert first == again


def test_the_interval_brackets_the_point_estimate() -> None:
    point = agreement.raw_agreement(WORKED)
    interval = agreement.bootstrap_ci(WORKED, agreement.raw_agreement, resamples=500)

    assert point is not None and interval is not None
    low, high = interval
    assert low <= point <= high


def test_no_interval_from_fewer_than_two_pairs() -> None:
    assert agreement.bootstrap_ci([("0", "0")], agreement.raw_agreement) is None


def test_confusion_and_marginals_count_what_occurred() -> None:
    counts = agreement.confusion(WORKED)
    human, judge = agreement.marginals(WORKED)

    assert counts[("0", "0")] == 3
    assert counts[("2", "0")] == 1
    assert human == {"0": 5, "1": 3, "2": 2}
    assert judge == {"0": 4, "1": 4, "2": 2}

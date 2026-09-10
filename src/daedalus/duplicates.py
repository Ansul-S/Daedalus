"""The duplicate safety net from `docs/PHASE-6-PROTOCOL.md` section 11.

Diversity in this benchmark is structural first: one question per section, and
no section generated from twice, so two questions cannot share a context by
construction. Similarity is measured **after the fact** to check that claim, not
to filter anything, and it takes no part in structural coverage.

Three properties of this module exist to keep the check honest.

**The bands are fixed on the cosine range, not on the observed distribution.**
Section 11 forbids choosing a threshold in advance and forbids choosing one for a
flattering rate. Strata chosen after seeing where the mass fell would be the same
mistake wearing a different hat: quantile bands would guarantee a populated tail
whether or not one exists, and hand-picked cuts could be moved until the answer
improved. These cuts are stated here, before the vectors were computed, and are
the same cuts whatever the data turns out to look like.

**The sample is drawn by seeded hash, not by shuffling.** A pair's place in the
sample depends only on its identity and the seed, so a redraw reproduces the same
pairs and the sample is not a free parameter.

**A band with fewer pairs than asked for is reported short, never padded.** If
the high-similarity tail holds three pairs, all three are labelled and the band
is reported as three. Borrowing from a neighbouring band to reach a round number
would misdescribe what the corpus contains, which is the one thing this check
exists to find out.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

#: The embedding identity for question vectors. Distinct from the production
#: `bge-m3` chunk embeddings, which section 11 step 1 requires be left untouched.
QUESTION_EMBEDDING_MODEL = "bge-m3-questions"

#: Seed for the stratified draw, so a redraw reproduces the same pairs.
SAMPLE_SEED = "phase6-duplicates-20260910"

#: How many pairs are drawn from each band.
PER_BAND = 15

#: Band lower bounds, fixed on the cosine range before any similarity was
#: computed. A pair belongs to the highest band whose bound it meets.
BAND_BOUNDS: tuple[tuple[str, float], ...] = (
    ("[0.95, 1.0]", 0.95),
    ("[0.90, 0.95)", 0.90),
    ("[0.80, 0.90)", 0.80),
    ("[0.70, 0.80)", 0.70),
    ("[0.60, 0.70)", 0.60),
    ("[0.50, 0.60)", 0.50),
    ("< 0.50", float("-inf")),
)

#: The bands in descending order, which is the order they are reported in.
BANDS: tuple[str, ...] = tuple(name for name, _bound in BAND_BOUNDS)


def band_for(similarity: float) -> str:
    """Return the band a similarity falls in."""
    for name, bound in BAND_BOUNDS:
        if similarity >= bound:
            return name
    raise ValueError(f"no band for similarity {similarity!r}")


def band_counts(pairs: Sequence[tuple[int, int, float]]) -> dict[str, int]:
    """Return how many pairs fall in each band, bands with none included."""
    counts = dict.fromkeys(BANDS, 0)
    for _lower, _higher, similarity in pairs:
        counts[band_for(similarity)] += 1
    return counts


def _draw_key(lower: int, higher: int, seed: str) -> str:
    return hashlib.md5(f"{lower}:{higher}:{seed}".encode()).hexdigest()


def sample_pairs(
    pairs: Sequence[tuple[int, int, float]],
    per_band: int = PER_BAND,
    seed: str = SAMPLE_SEED,
) -> list[tuple[int, int, float, str]]:
    """Draw a stratified sample, returning (lower, higher, similarity, band).

    Within each band pairs are ordered by a hash of their identity and the seed,
    and the first `per_band` are taken. A band holding fewer than that
    contributes all of them. The result is ordered by band, highest first, then
    by similarity descending, so the tail is presented first.
    """
    if per_band < 1:
        raise ValueError(f"per_band must be 1 or more, got {per_band}")

    grouped: dict[str, list[tuple[int, int, float]]] = {band: [] for band in BANDS}
    for pair in pairs:
        grouped[band_for(pair[2])].append(pair)

    drawn: list[tuple[int, int, float, str]] = []
    for band in BANDS:
        members = sorted(grouped[band], key=lambda p: _draw_key(p[0], p[1], seed))[
            :per_band
        ]
        drawn.extend(
            sorted(
                ((lower, higher, sim, band) for lower, higher, sim in members),
                key=lambda p: -p[2],
            )
        )
    return drawn


def separation(labels: Sequence[tuple[float, str]], threshold: float) -> dict[str, int]:
    """Count how a threshold classifies the labelled pairs.

    `labels` is (similarity, value) with value "duplicate" or "not_duplicate".
    A pair at or above the threshold is predicted duplicate.
    """
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for similarity, value in labels:
        predicted = similarity >= threshold
        actual = value == "duplicate"
        if predicted and actual:
            counts["tp"] += 1
        elif predicted and not actual:
            counts["fp"] += 1
        elif not predicted and actual:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
    return counts


def best_threshold(
    labels: Sequence[tuple[float, str]],
) -> tuple[float, dict[str, int]] | None:
    """Return the threshold separating the labelled pairs best, and its counts.

    Candidates are the midpoints between adjacent observed similarities, so the
    search cannot land on a value no pair could distinguish. "Best" is the
    highest count of correct classifications, ties broken towards the higher
    threshold, which is the more conservative duplicate claim.

    Returns None when the labels contain only one class, where separation is not
    a question that has an answer.
    """
    values = {value for _similarity, value in labels}
    if len(values) < 2:
        return None

    ordered = sorted({similarity for similarity, _value in labels})
    candidates = [
        (low + high) / 2 for low, high in zip(ordered, ordered[1:], strict=False)
    ]
    if not candidates:
        return None

    best: tuple[float, dict[str, int]] | None = None
    best_correct = -1
    for threshold in candidates:
        counts = separation(labels, threshold)
        correct = counts["tp"] + counts["tn"]
        if correct > best_correct or (
            correct == best_correct and best is not None and threshold > best[0]
        ):
            best_correct = correct
            best = (threshold, counts)
    return best

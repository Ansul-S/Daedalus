"""
The labelled retrieval dataset.

One JSON object per line, described in docs/pipelines/EVALUATION_ENGINE.md.
JSONL rather than a single JSON array so the file diffs cleanly: labelling
is incremental and a reviewer needs to see one query change, not a
reindented document.

Most of this module is validation, which is deliberate. A benchmark whose
labels are subtly wrong is worse than no benchmark, because it produces
confident numbers that cannot be trusted and nobody notices. The check
that matters most is ``verify_anchors``: it re-reads each quote out of the
frozen text at the offsets the label claims, so a label that has drifted
is a loud failure rather than a slightly depressed recall score.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from daedalus.config import constants
from daedalus.evaluation.corpus import FrozenDocument
from daedalus.evaluation.spans import RelevantSpan

__all__ = [
    "EvalQuery",
    "LabelledSpan",
    "load_dataset",
    "verify_anchors",
    "write_dataset",
]


logger = logging.getLogger(__name__)


class LabelledSpan(BaseModel):
    """A character range in a document's frozen parsed text."""

    doc_id: str
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)

    # Stored so a human can check the label without opening the source, and
    # so drift is detectable — see verify_anchors.
    quote: str = Field(min_length=1)

    grade: int = Field(default=constants.GRADE_ESSENTIAL, ge=1, le=2)

    @model_validator(mode="after")
    def _check_range(self) -> LabelledSpan:
        if self.char_end <= self.char_start:
            raise ValueError(f"{self.doc_id}: char_end must be greater than char_start")

        return self

    def as_span(self) -> RelevantSpan:
        return RelevantSpan(
            doc_id=self.doc_id,
            char_start=self.char_start,
            char_end=self.char_end,
            grade=self.grade,
        )


class EvalQuery(BaseModel):
    """One benchmark query and its labels."""

    id: str
    query: str = Field(min_length=1)
    query_type: str
    source_type: str
    answerable: bool = True
    paraphrased: bool = False
    split: str = "dev"
    relevant_spans: list[LabelledSpan] = Field(default_factory=list)
    reference_answer: str = ""
    verified_by: str = "unverified"
    notes: str = ""

    @model_validator(mode="after")
    def _check_consistency(self) -> EvalQuery:
        if self.query_type not in constants.QUERY_TYPES:
            raise ValueError(
                f"{self.id}: query_type {self.query_type!r} not in {sorted(constants.QUERY_TYPES)}"
            )

        if self.split not in constants.SPLITS:
            raise ValueError(f"{self.id}: split {self.split!r} not in {sorted(constants.SPLITS)}")

        # The two ways this can be incoherent are both silent disasters: an
        # answerable query with no labels scores zero recall forever, and an
        # unanswerable one with labels is not measuring refusal at all.
        if self.answerable and not self.relevant_spans:
            raise ValueError(f"{self.id}: answerable query has no relevant spans")

        if not self.answerable and self.relevant_spans:
            raise ValueError(f"{self.id}: unanswerable query has relevant spans")

        return self

    def spans(self) -> list[RelevantSpan]:
        return [span.as_span() for span in self.relevant_spans]


def load_dataset(path: Path | None = None) -> list[EvalQuery]:
    """Read and validate a JSONL dataset."""

    target = path or constants.EVAL_DATASETS_DIR / "retrieval.jsonl"

    if not target.exists():
        raise FileNotFoundError(f"no dataset at {target}")

    queries: list[EvalQuery] = []
    seen: set[str] = set()

    for number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue

        try:
            query = EvalQuery.model_validate(json.loads(line))
        except Exception as error:
            raise ValueError(f"{target.name} line {number}: {error}") from error

        # Duplicate ids would double-count a query in every slice it appears
        # in, which is invisible in the output.
        if query.id in seen:
            raise ValueError(f"{target.name} line {number}: duplicate id {query.id!r}")

        seen.add(query.id)
        queries.append(query)

    return queries


def write_dataset(queries: Iterable[EvalQuery], path: Path) -> int:
    """Write a dataset as JSONL, one query per line."""

    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [query.model_dump_json() for query in queries]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return len(lines)


def verify_anchors(queries: Sequence[EvalQuery], documents: Sequence[FrozenDocument]) -> list[str]:
    """
    Check every label still points at the text it claims.

    Returns a list of human-readable problems, empty when the dataset is
    sound. Catches the failure that matters: frozen text was re-generated,
    every offset downstream shifted, and the labels now point at
    neighbouring material while still scoring plausibly.

    The check is a substring test, so it has slack: a quote much shorter
    than its span survives a shift that leaves it anywhere inside that
    span. Sensitivity rises as the quote approaches the span in length.
    """

    by_id = {document.doc_id: document for document in documents}
    problems: list[str] = []

    for query in queries:
        for span in query.relevant_spans:
            document = by_id.get(span.doc_id)

            if document is None:
                problems.append(f"{query.id}: no frozen document {span.doc_id!r}")
                continue

            if span.char_end > len(document.text):
                problems.append(
                    f"{query.id}: span {span.char_start}-{span.char_end} runs past "
                    f"{span.doc_id} ({len(document.text)} chars)"
                )
                continue

            anchored = document.text[span.char_start : span.char_end]

            # Substring rather than equality: a quote is allowed to be the
            # memorable fragment of a longer relevant span.
            if span.quote not in anchored:
                problems.append(
                    f"{query.id}: quote not found at {span.doc_id}"
                    f":{span.char_start}-{span.char_end}"
                )

    return problems

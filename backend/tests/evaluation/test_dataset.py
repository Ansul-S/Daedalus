"""Dataset loading, validation, and anchor verification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from daedalus.evaluation.corpus import FrozenDocument
from daedalus.evaluation.dataset import EvalQuery, load_dataset, verify_anchors, write_dataset

TEXT = "Attention scales the dot products by one over the square root of d_k."


def frozen(text: str = TEXT) -> FrozenDocument:
    return FrozenDocument(
        doc_id="paper",
        filename="paper.pdf",
        source_type="arxiv",
        text=text,
        segments=(),
        n_pages=1,
        text_sha256="unused",
    )


def query(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "id": "ret-0001",
        "query": "why scale?",
        "query_type": "conceptual",
        "source_type": "arxiv",
        "answerable": True,
        "split": "dev",
        "relevant_spans": [
            {
                "doc_id": "paper",
                "char_start": 10,
                "char_end": 40,
                "quote": "scales the dot",
                "grade": 2,
            }
        ],
    }
    record.update(overrides)

    return record


def write(tmp_path: Path, *records: dict[str, object]) -> Path:
    path = tmp_path / "retrieval.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    return path


# Loading


def test_a_valid_dataset_loads(tmp_path: Path) -> None:
    queries = load_dataset(write(tmp_path, query()))

    assert [item.id for item in queries] == ["ret-0001"]


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.jsonl"
    path.write_text(json.dumps(query()) + "\n\n", encoding="utf-8")

    assert len(load_dataset(path)) == 1


def test_a_missing_dataset_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "absent.jsonl")


def test_the_failing_line_number_is_reported(tmp_path: Path) -> None:
    path = write(tmp_path, query(), query(id="ret-0002", query_type="nonsense"))

    with pytest.raises(ValueError, match="line 2"):
        load_dataset(path)


def test_a_duplicate_id_is_refused(tmp_path: Path) -> None:
    """A repeated id would be double-counted in every slice, invisibly."""

    with pytest.raises(ValueError, match="duplicate id"):
        load_dataset(write(tmp_path, query(), query()))


# Schema


def test_an_unknown_query_type_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="query_type"):
        load_dataset(write(tmp_path, query(query_type="vibes")))


def test_an_unknown_split_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="split"):
        load_dataset(write(tmp_path, query(split="train")))


def test_an_answerable_query_without_labels_is_refused(tmp_path: Path) -> None:
    """It would score zero recall forever, and look like a retrieval failure."""

    with pytest.raises(ValueError, match="no relevant spans"):
        load_dataset(write(tmp_path, query(relevant_spans=[])))


def test_an_unanswerable_query_with_labels_is_refused(tmp_path: Path) -> None:
    """It would not be measuring refusal at all."""

    with pytest.raises(ValueError, match="has relevant spans"):
        load_dataset(write(tmp_path, query(answerable=False)))


def test_an_unanswerable_query_without_labels_is_accepted(tmp_path: Path) -> None:
    path = write(tmp_path, query(answerable=False, query_type="unanswerable", relevant_spans=[]))

    assert load_dataset(path)[0].answerable is False


def test_a_reversed_span_is_refused(tmp_path: Path) -> None:
    span = {"doc_id": "paper", "char_start": 40, "char_end": 10, "quote": "x", "grade": 2}

    with pytest.raises(ValueError, match="char_end"):
        load_dataset(write(tmp_path, query(relevant_spans=[span])))


def test_a_grade_outside_the_scale_is_refused(tmp_path: Path) -> None:
    span = {"doc_id": "paper", "char_start": 0, "char_end": 10, "quote": "x", "grade": 5}

    with pytest.raises(ValueError):
        load_dataset(write(tmp_path, query(relevant_spans=[span])))


# Anchors


def test_a_correctly_anchored_label_verifies() -> None:
    queries = [EvalQuery.model_validate(query())]

    assert verify_anchors(queries, [frozen()]) == []


def test_a_quote_that_moved_is_caught() -> None:
    """The failure this exists for: re-parsing shifted every offset."""

    queries = [EvalQuery.model_validate(query())]
    shifted = frozen("A" * 40 + TEXT)

    problems = verify_anchors(queries, [shifted])

    assert len(problems) == 1
    assert "quote not found" in problems[0]


def test_a_shift_smaller_than_the_span_slack_is_not_caught() -> None:
    """A documented limit, not a bug.

    The quote is a fragment of a wider span, so a shift small enough to
    leave the quote inside that span still verifies. Detection improves as
    the quote approaches the span in length.
    """

    queries = [EvalQuery.model_validate(query())]

    assert verify_anchors(queries, [frozen("A" * 5 + TEXT)]) == []


def test_a_span_running_past_the_document_is_caught() -> None:
    span = {
        "doc_id": "paper",
        "char_start": 0,
        "char_end": 99_999,
        "quote": "Attention",
        "grade": 2,
    }
    queries = [EvalQuery.model_validate(query(relevant_spans=[span]))]

    assert "runs past" in verify_anchors(queries, [frozen()])[0]


def test_a_label_naming_an_unfrozen_document_is_caught() -> None:
    span = {"doc_id": "ghost", "char_start": 0, "char_end": 10, "quote": "x", "grade": 2}
    queries = [EvalQuery.model_validate(query(relevant_spans=[span]))]

    assert "no frozen document" in verify_anchors(queries, [frozen()])[0]


# Round trip


def test_a_dataset_survives_a_write_and_read(tmp_path: Path) -> None:
    original = load_dataset(write(tmp_path, query()))
    path = tmp_path / "out.jsonl"

    write_dataset(original, path)

    assert [item.id for item in load_dataset(path)] == [item.id for item in original]

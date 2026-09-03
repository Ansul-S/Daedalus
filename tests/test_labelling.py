"""Tests for reference-set storage and the labelling loop.

The loop reads keystrokes from stdin, which is replaced here, so no terminal
interaction is required. The embedder is stubbed.
"""

from __future__ import annotations

import io
from pathlib import Path

import psycopg
import pytest

from daedalus import cli
from daedalus.document import Document, Segment, SegmentKind
from daedalus.storage.database import DATABASE_URL_ENV
from daedalus.storage.documents import store_document
from daedalus.storage.queries import (
    add_query,
    grade_totals,
    judged_pairs,
    list_queries,
    record_judgement,
)

Connection = psycopg.Connection[tuple[object, ...]]

TEXTS = [
    "the reader model predicts a start and end span over the passage",
    "vector databases store embeddings and support similarity search",
    "bagging reduces variance by averaging many decision trees",
]


def seed_corpus(connection: Connection) -> None:
    store_document(
        connection,
        Document(
            doc_id="d1",
            source_path=Path("/x.ipynb"),
            source_format="notebook",
            title="T",
            segments=tuple(
                Segment(i, SegmentKind.PROSE, text, ("A",), (), f"cell:{i}")
                for i, text in enumerate(TEXTS)
            ),
        ),
    )


def test_add_query_returns_an_id(connection: Connection) -> None:
    assert add_query(connection, "what is bagging", "authored") is not None


def test_add_query_is_idempotent_on_text(connection: Connection) -> None:
    first = add_query(connection, "what is bagging", "authored")
    again = add_query(connection, "what is bagging", "harvested")

    assert first is not None
    assert again is None
    assert len(list_queries(connection)) == 1


def test_add_query_rejects_an_unknown_source(connection: Connection) -> None:
    with pytest.raises(ValueError, match="source must be one of"):
        add_query(connection, "q", "invented")


def test_record_judgement_rejects_a_bad_grade(connection: Connection) -> None:
    seed_corpus(connection)
    query_id = add_query(connection, "q", "authored")
    assert query_id is not None

    with pytest.raises(ValueError, match="grade must be one of"):
        record_judgement(connection, query_id, "d1", 0, 5)


def test_record_judgement_replaces_an_earlier_grade(connection: Connection) -> None:
    seed_corpus(connection)
    query_id = add_query(connection, "q", "authored")
    assert query_id is not None

    record_judgement(connection, query_id, "d1", 0, 1)
    record_judgement(connection, query_id, "d1", 0, 2)

    assert grade_totals(connection) == {2: 1}


def test_judged_pairs_tracks_progress(connection: Connection) -> None:
    seed_corpus(connection)
    query_id = add_query(connection, "q", "authored")
    assert query_id is not None
    record_judgement(connection, query_id, "d1", 1, 0)

    assert judged_pairs(connection, query_id) == {("d1", 1)}


def test_list_queries_counts_judgements(connection: Connection) -> None:
    seed_corpus(connection)
    query_id = add_query(connection, "q", "authored")
    assert query_id is not None
    record_judgement(connection, query_id, "d1", 0, 2)
    record_judgement(connection, query_id, "d1", 1, 0)

    assert [q.judged for q in list_queries(connection)] == [2]


def drive(monkeypatch: pytest.MonkeyPatch, keys: str) -> None:
    """Feed the labelling loop a scripted sequence of keystrokes."""
    monkeypatch.setattr("sys.stdin", io.StringIO("\n".join(keys) + "\n"))


def prepare(
    connection: Connection, database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(DATABASE_URL_ENV, database_url)
    monkeypatch.setattr(
        cli, "embed_texts", lambda texts, model="m": [[1.0, 0.0] for _ in texts]
    )
    seed_corpus(connection)
    connection.commit()


def test_label_records_grades(
    connection: Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare(connection, database_url, monkeypatch)
    add_query(connection, "reader embeddings trees", "authored")
    connection.commit()
    drive(monkeypatch, "210")

    assert (
        cli.main(["label", "--vector-k", "0", "--lexical-k", "3", "--random-k", "0"])
        == 0
    )

    assert sum(grade_totals(connection).values()) == 3
    assert "judgements by grade" in capsys.readouterr().out


def test_label_quits_on_q(
    connection: Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare(connection, database_url, monkeypatch)
    add_query(connection, "reader embeddings trees", "authored")
    connection.commit()
    drive(monkeypatch, "2q1")

    assert (
        cli.main(["label", "--vector-k", "0", "--lexical-k", "3", "--random-k", "0"])
        == 0
    )

    assert sum(grade_totals(connection).values()) == 1
    assert "stopped" in capsys.readouterr().out


def test_label_reprompts_on_an_unrecognised_key(
    connection: Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A stray key re-offers the same candidate instead of advancing past it."""
    prepare(connection, database_url, monkeypatch)
    add_query(connection, "reader embeddings trees", "authored")
    connection.commit()
    drive(monkeypatch, "x2")

    assert (
        cli.main(["label", "--vector-k", "0", "--lexical-k", "1", "--random-k", "0"])
        == 0
    )

    assert sum(grade_totals(connection).values()) == 1
    assert "not one of" in capsys.readouterr().out


def test_label_stops_when_input_ends(
    connection: Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exhausted input ends the session rather than skipping what remains."""
    prepare(connection, database_url, monkeypatch)
    add_query(connection, "reader embeddings trees", "authored")
    connection.commit()
    drive(monkeypatch, "")

    assert (
        cli.main(["label", "--vector-k", "0", "--lexical-k", "2", "--random-k", "0"])
        == 0
    )

    assert sum(grade_totals(connection).values()) == 0
    assert "input ended" in capsys.readouterr().out


def test_label_skips_already_judged_candidates(
    connection: Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare(connection, database_url, monkeypatch)
    query_id = add_query(connection, "reader embeddings trees", "authored")
    assert query_id is not None
    connection.commit()

    drive(monkeypatch, "2")
    cli.main(
        [
            "label",
            "--vector-k",
            "0",
            "--lexical-k",
            "3",
            "--random-k",
            "0",
            "--per-query",
            "1",
        ]
    )
    capsys.readouterr()
    first = judged_pairs(connection, query_id)

    drive(monkeypatch, "11")
    cli.main(["label", "--vector-k", "0", "--lexical-k", "3", "--random-k", "0"])

    remaining = judged_pairs(connection, query_id) - first
    assert len(remaining) == 2


def test_label_reports_nothing_to_do(
    connection: Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare(connection, database_url, monkeypatch)

    assert cli.main(["label"]) == 0
    assert "nothing to label" in capsys.readouterr().out


def test_query_add_and_list_commands(
    connection: Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(DATABASE_URL_ENV, database_url)

    assert cli.main(["query", "add", "what is bagging", "--source", "authored"]) == 0
    assert "added query" in capsys.readouterr().out

    assert cli.main(["query", "add", "what is bagging", "--source", "authored"]) == 0
    assert "already present" in capsys.readouterr().out

    assert cli.main(["query", "list"]) == 0
    output = capsys.readouterr().out
    assert "what is bagging" in output
    assert "authored" in output


def test_policy_is_shown_at_session_start(
    connection: Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare(connection, database_url, monkeypatch)
    add_query(connection, "reader embeddings trees", "authored")
    connection.commit()
    drive(monkeypatch, "q")

    cli.main(["label", "--vector-k", "0", "--lexical-k", "1", "--random-k", "0"])
    output = capsys.readouterr().out

    assert "0  not relevant" in output
    assert "1  partially answers" in output
    assert "2  fully answers" in output
    assert "docs/LABELLING.md" in output


def test_grade_meanings_appear_on_every_prompt(
    connection: Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare(connection, database_url, monkeypatch)
    add_query(connection, "reader embeddings trees", "authored")
    connection.commit()
    drive(monkeypatch, "22")

    cli.main(["label", "--vector-k", "0", "--lexical-k", "2", "--random-k", "0"])
    output = capsys.readouterr().out

    assert output.count("0 not relevant") >= 2
    assert output.count("2 fully answers") >= 2


def test_retriever_provenance_is_never_shown(
    connection: Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A judgement influenced by the retriever would be measuring itself."""
    prepare(connection, database_url, monkeypatch)
    add_query(connection, "reader embeddings trees", "authored")
    connection.commit()
    drive(monkeypatch, "2q")

    cli.main(["label", "--vector-k", "1", "--lexical-k", "1", "--random-k", "1"])
    output = capsys.readouterr().out.lower()

    assert "vector" not in output.replace("vector databases", "")
    assert "lexical" not in output
    assert "rank" not in output
    assert "score" not in output


def test_policy_key_reprints_without_grading(
    connection: Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare(connection, database_url, monkeypatch)
    add_query(connection, "reader embeddings trees", "authored")
    connection.commit()
    drive(monkeypatch, "?2")

    cli.main(["label", "--vector-k", "0", "--lexical-k", "1", "--random-k", "0"])

    assert sum(grade_totals(connection).values()) == 1


def test_full_text_key_shows_untruncated_content(
    connection: Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(DATABASE_URL_ENV, database_url)
    monkeypatch.setattr(
        cli, "embed_texts", lambda texts, model="m": [[1.0, 0.0] for _ in texts]
    )
    long_text = "reader " + ("x" * cli.PREVIEW_LIMIT) + " ENDMARKER"
    store_document(
        connection,
        Document(
            doc_id="d2",
            source_path=Path("/y.ipynb"),
            source_format="notebook",
            title="T",
            segments=(Segment(0, SegmentKind.PROSE, long_text, ("A",), (), "cell:0"),),
        ),
    )
    connection.commit()
    add_query(connection, "reader", "authored")
    connection.commit()
    drive(monkeypatch, "f2")

    cli.main(["label", "--vector-k", "0", "--lexical-k", "1", "--random-k", "0"])
    output = capsys.readouterr().out

    assert "[TRUNCATED" in output
    assert "ENDMARKER" in output
    assert sum(grade_totals(connection).values()) == 1


def test_skipping_records_nothing(
    connection: Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare(connection, database_url, monkeypatch)
    add_query(connection, "reader embeddings trees", "authored")
    connection.commit()
    drive(monkeypatch, "sss")

    cli.main(["label", "--vector-k", "0", "--lexical-k", "3", "--random-k", "0"])

    assert grade_totals(connection) == {}


def test_skipped_candidates_are_offered_again(
    connection: Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare(connection, database_url, monkeypatch)
    add_query(connection, "reader embeddings trees", "authored")
    connection.commit()

    drive(monkeypatch, "sss")
    cli.main(["label", "--vector-k", "0", "--lexical-k", "3", "--random-k", "0"])
    capsys.readouterr()

    drive(monkeypatch, "222")
    cli.main(["label", "--vector-k", "0", "--lexical-k", "3", "--random-k", "0"])

    assert grade_totals(connection) == {2: 3}


def test_no_grade_is_recorded_without_a_keystroke(
    connection: Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Grades must be human judgements, never derived from retrieval."""
    prepare(connection, database_url, monkeypatch)
    add_query(connection, "reader embeddings trees", "authored")
    connection.commit()
    drive(monkeypatch, "q")

    cli.main(["label", "--vector-k", "3", "--lexical-k", "3", "--random-k", "3"])

    assert grade_totals(connection) == {}


def test_preview_limit_is_two_thousand() -> None:
    """Chosen against the corpus so most chunks display whole."""
    assert cli.PREVIEW_LIMIT == 2000


def test_truncation_is_marked_explicitly(
    connection: Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(DATABASE_URL_ENV, database_url)
    monkeypatch.setattr(
        cli, "embed_texts", lambda texts, model="m": [[1.0, 0.0] for _ in texts]
    )
    store_document(
        connection,
        Document(
            doc_id="d3",
            source_path=Path("/z.ipynb"),
            source_format="notebook",
            title="T",
            segments=(
                Segment(
                    0,
                    SegmentKind.PROSE,
                    "reader " + ("x" * (cli.PREVIEW_LIMIT + 500)),
                    ("A",),
                    (),
                    "cell:0",
                ),
            ),
        ),
    )
    connection.commit()
    add_query(connection, "reader", "authored")
    connection.commit()
    drive(monkeypatch, "2")

    cli.main(["label", "--vector-k", "0", "--lexical-k", "1", "--random-k", "0"])
    output = capsys.readouterr().out

    assert "[TRUNCATED" in output
    assert "Press f to read all before judging." in output


def seed_with_output(connection: Connection) -> None:
    """A code chunk and the output it produced."""
    store_document(
        connection,
        Document(
            doc_id="d4",
            source_path=Path("/w.ipynb"),
            source_format="notebook",
            title="T",
            segments=(
                Segment(
                    0,
                    SegmentKind.CODE,
                    "retriever = build_retriever(embeddings)",
                    ("A",),
                    (),
                    "cell:0",
                ),
                Segment(
                    1,
                    SegmentKind.OUTPUT,
                    "retriever ready: 5 chunks",
                    ("A",),
                    (),
                    "cell:0",
                    parent_ordinal=0,
                ),
            ),
        ),
    )


def test_output_chunk_shows_parent_as_context(
    connection: Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(DATABASE_URL_ENV, database_url)
    monkeypatch.setattr(
        cli, "embed_texts", lambda texts, model="m": [[1.0, 0.0] for _ in texts]
    )
    seed_with_output(connection)
    connection.commit()
    add_query(connection, "retriever ready chunks", "authored")
    connection.commit()
    drive(monkeypatch, "22")

    cli.main(["label", "--vector-k", "0", "--lexical-k", "2", "--random-k", "0"])
    output = capsys.readouterr().out

    assert "CONTEXT — the code that produced this output. NOT judged." in output
    assert "CANDIDATE — judge this:" in output
    assert "build_retriever" in output


def test_prose_chunk_shows_no_context_block(
    connection: Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare(connection, database_url, monkeypatch)
    add_query(connection, "reader embeddings trees", "authored")
    connection.commit()
    drive(monkeypatch, "2q")

    cli.main(["label", "--vector-k", "0", "--lexical-k", "1", "--random-k", "0"])
    output = capsys.readouterr().out

    assert "CONTEXT" not in output
    assert "CANDIDATE" not in output


def test_grade_is_recorded_against_the_output_not_the_parent(
    connection: Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Context must not shift the judgement onto the parent chunk."""
    monkeypatch.setenv(DATABASE_URL_ENV, database_url)
    monkeypatch.setattr(
        cli, "embed_texts", lambda texts, model="m": [[1.0, 0.0] for _ in texts]
    )
    seed_with_output(connection)
    connection.commit()
    query_id = add_query(connection, "retriever ready chunks", "authored")
    assert query_id is not None
    connection.commit()
    drive(monkeypatch, "2s")

    cli.main(["label", "--vector-k", "0", "--lexical-k", "2", "--random-k", "0"])

    judged = judged_pairs(connection, query_id)
    assert len(judged) == 1
    assert judged.issubset({("d4", 0), ("d4", 1)})


def test_pooling_defaults_are_unchanged() -> None:
    args = cli.build_parser().parse_args(["label"])
    assert (args.vector_k, args.lexical_k, args.random_k, args.per_query) == (
        10,
        10,
        5,
        25,
    )


def test_only_three_grades_are_accepted() -> None:
    assert cli.GRADE_KEYS == {"0": 0, "1": 1, "2": 2}


def test_per_query_cap_leaves_candidates_unjudged_not_zero(
    connection: Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A capped query leaves candidates absent from the set, not graded 0."""
    prepare(connection, database_url, monkeypatch)
    query_id = add_query(connection, "reader embeddings trees", "authored")
    assert query_id is not None
    connection.commit()
    drive(monkeypatch, "2")

    cli.main(
        [
            "label",
            "--vector-k",
            "0",
            "--lexical-k",
            "3",
            "--random-k",
            "0",
            "--per-query",
            "1",
        ]
    )

    assert len(judged_pairs(connection, query_id)) == 1
    assert grade_totals(connection) == {2: 1}


def test_regrade_offers_judged_candidates_again(
    connection: Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare(connection, database_url, monkeypatch)
    query_id = add_query(connection, "reader embeddings trees", "authored")
    assert query_id is not None
    connection.commit()

    drive(monkeypatch, "000")
    cli.main(["label", "--vector-k", "0", "--lexical-k", "3", "--random-k", "0"])
    capsys.readouterr()
    assert grade_totals(connection) == {0: 3}

    drive(monkeypatch, "222")
    cli.main(
        [
            "label",
            "--vector-k",
            "0",
            "--lexical-k",
            "3",
            "--random-k",
            "0",
            "--regrade",
            str(query_id),
        ]
    )
    capsys.readouterr()

    assert grade_totals(connection) == {2: 3}
    assert len(judged_pairs(connection, query_id)) == 3


def test_regrade_ignores_the_per_query_ceiling(
    connection: Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare(connection, database_url, monkeypatch)
    query_id = add_query(connection, "reader embeddings trees", "authored")
    assert query_id is not None
    connection.commit()

    drive(monkeypatch, "000")
    cli.main(["label", "--vector-k", "0", "--lexical-k", "3", "--random-k", "0"])
    capsys.readouterr()

    drive(monkeypatch, "111")
    cli.main(
        [
            "label",
            "--vector-k",
            "0",
            "--lexical-k",
            "3",
            "--random-k",
            "0",
            "--per-query",
            "1",
            "--regrade",
            str(query_id),
        ]
    )
    capsys.readouterr()

    assert grade_totals(connection) == {1: 3}


def test_regrade_rejects_an_unknown_query_id(
    connection: Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare(connection, database_url, monkeypatch)
    connection.commit()

    assert cli.main(["label", "--regrade", "9999"]) == 1
    assert "no query with id 9999" in capsys.readouterr().out

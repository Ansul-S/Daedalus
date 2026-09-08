"""Tests for the command line entry point.

Path expansion and argument parsing need no database. The command tests use the
throwaway database and a stubbed embedder, so no model is contacted.
"""

from __future__ import annotations

from pathlib import Path

import psycopg
import pytest

from daedalus import cli
from daedalus.document import Document, Segment, SegmentKind
from daedalus.storage.database import DATABASE_URL_ENV
from daedalus.storage.documents import chunks_at, store_document
from daedalus.storage.questions import (
    GeneratedQuestion,
    failure_mode_counts,
    label_totals,
    list_questions,
    record_questions,
)
from tests.test_notebook import code, md, write_notebook

Connection = psycopg.Connection[tuple[object, ...]]


def make_notebook(path: Path, title: str = "A") -> Path:
    return write_notebook(path, [md(f"# {title}"), code("x = 1")])


def test_notebook_paths_expands_a_directory(tmp_path: Path) -> None:
    make_notebook(tmp_path / "b.ipynb")
    make_notebook(tmp_path / "a.ipynb")
    (tmp_path / "notes.txt").write_text("ignored")

    found = cli.notebook_paths([tmp_path])

    assert [p.name for p in found] == ["a.ipynb", "b.ipynb"]


def test_notebook_paths_accepts_a_file(tmp_path: Path) -> None:
    path = make_notebook(tmp_path / "one.ipynb")
    assert cli.notebook_paths([path]) == [path]


def test_notebook_paths_rejects_other_files(tmp_path: Path) -> None:
    other = tmp_path / "notes.txt"
    other.write_text("x")

    with pytest.raises(ValueError, match="not a notebook"):
        cli.notebook_paths([other])


def test_parser_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_ingest_force_defaults_off() -> None:
    assert cli.build_parser().parse_args(["ingest", "x.ipynb"]).force is False
    assert cli.build_parser().parse_args(["ingest", "x.ipynb", "--force"]).force is True


def test_embed_defaults(tmp_path: Path) -> None:
    args = cli.build_parser().parse_args(["embed"])
    assert args.model == "bge-m3"
    assert args.batch_size == 32


def test_embed_accepts_overrides() -> None:
    args = cli.build_parser().parse_args(
        ["embed", "--model", "other", "--batch-size", "8"]
    )
    assert (args.model, args.batch_size) == ("other", 8)


def test_missing_configuration_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(DATABASE_URL_ENV, raising=False)
    make_notebook(tmp_path / "a.ipynb")

    assert cli.main(["ingest", str(tmp_path)]) == 1
    assert DATABASE_URL_ENV in capsys.readouterr().err


def test_unreadable_path_is_reported(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert cli.main(["ingest", str(tmp_path / "notes.txt")]) == 1
    assert "not a notebook" in capsys.readouterr().err


def test_ingest_then_status(
    connection: Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(DATABASE_URL_ENV, database_url)
    make_notebook(tmp_path / "a.ipynb", title="First")
    make_notebook(tmp_path / "b.ipynb", title="Second")

    assert cli.main(["ingest", str(tmp_path)]) == 0
    ingest_output = capsys.readouterr().out
    assert "a.ipynb: 2 chunks" in ingest_output
    assert "b.ipynb: 2 chunks" in ingest_output

    assert cli.main(["status"]) == 0
    status_output = capsys.readouterr().out
    assert "documents: 2" in status_output
    assert "chunks:    4" in status_output
    assert "embeddings: none" in status_output


def test_embed_command_reports_progress(
    connection: Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(DATABASE_URL_ENV, database_url)
    monkeypatch.setattr(
        cli, "embed_texts", lambda texts, model="m": [[0.5, 0.5] for _ in texts]
    )
    make_notebook(tmp_path / "a.ipynb")
    cli.main(["ingest", str(tmp_path)])
    capsys.readouterr()

    assert cli.main(["embed", "--model", "stub"]) == 0
    assert "embedded 2 chunks with stub" in capsys.readouterr().out

    assert cli.main(["status"]) == 0
    assert "2 with stub (2 dimensions)" in capsys.readouterr().out


def test_embed_is_idempotent(
    connection: Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(DATABASE_URL_ENV, database_url)
    monkeypatch.setattr(
        cli, "embed_texts", lambda texts, model="m": [[1.0] for _ in texts]
    )
    make_notebook(tmp_path / "a.ipynb")
    cli.main(["ingest", str(tmp_path)])
    cli.main(["embed", "--model", "stub"])
    capsys.readouterr()

    assert cli.main(["embed", "--model", "stub"]) == 0
    assert "embedded 0 chunks" in capsys.readouterr().out


def test_reingesting_unchanged_material_is_skipped(
    connection: Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(DATABASE_URL_ENV, database_url)
    make_notebook(tmp_path / "a.ipynb")

    cli.main(["ingest", str(tmp_path)])
    capsys.readouterr()

    assert cli.main(["ingest", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "unchanged, skipped" in output
    assert "stored 0, skipped 1" in output


def test_skipped_reingest_preserves_embeddings(
    connection: Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The reason the skip exists: embeddings must survive a re-ingest."""
    monkeypatch.setenv(DATABASE_URL_ENV, database_url)
    monkeypatch.setattr(
        cli, "embed_texts", lambda texts, model="m": [[1.0] for _ in texts]
    )
    make_notebook(tmp_path / "a.ipynb")
    cli.main(["ingest", str(tmp_path)])
    cli.main(["embed", "--model", "stub"])
    capsys.readouterr()

    cli.main(["ingest", str(tmp_path)])

    assert cli.main(["embed", "--model", "stub"]) == 0
    assert "embedded 0 chunks" in capsys.readouterr().out


def test_force_restores_and_discards_embeddings(
    connection: Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(DATABASE_URL_ENV, database_url)
    monkeypatch.setattr(
        cli, "embed_texts", lambda texts, model="m": [[1.0] for _ in texts]
    )
    make_notebook(tmp_path / "a.ipynb")
    cli.main(["ingest", str(tmp_path)])
    cli.main(["embed", "--model", "stub"])
    capsys.readouterr()

    assert cli.main(["ingest", str(tmp_path), "--force"]) == 0
    assert "stored 1, skipped 0" in capsys.readouterr().out

    assert cli.main(["embed", "--model", "stub"]) == 0
    assert "embedded 2 chunks" in capsys.readouterr().out


def test_changed_material_is_not_skipped(
    connection: Connection,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(DATABASE_URL_ENV, database_url)
    path = tmp_path / "a.ipynb"
    make_notebook(path, title="First")
    cli.main(["ingest", str(path)])
    capsys.readouterr()

    make_notebook(path, title="Second")
    assert cli.main(["ingest", str(path)]) == 0
    assert "stored 1, skipped 0" in capsys.readouterr().out


# --- question labelling -------------------------------------------------


def store_two_questions(connection: Connection) -> None:
    """Two accepted questions in one document, each citing its own seed."""
    store_document(
        connection,
        Document(
            doc_id="d1",
            source_path=Path("/corpus/n.ipynb"),
            source_format="notebook",
            title="n",
            segments=(
                Segment(0, SegmentKind.PROSE, "Bagging averages.", ("S1",), (), "c:0"),
                Segment(1, SegmentKind.PROSE, "Trees overfit.", ("S2",), (), "c:1"),
            ),
        ),
    )
    record_questions(
        connection,
        [
            GeneratedQuestion(
                doc_id="d1",
                heading_path=("S1",),
                seed_ordinal=0,
                selection_rank=1,
                requested_type="conceptual",
                requested_difficulty="hard",
                text="Why does bagging reduce variance?",
                grounding_quote="Bagging averages.",
                context_ordinals=(0,),
                cited_ordinals=(0,),
                model="qwen3:8b",
                prompt_version="p6-v2",
                params_hash="abc",
            ),
            GeneratedQuestion(
                doc_id="d1",
                heading_path=("S2",),
                seed_ordinal=1,
                selection_rank=2,
                requested_type="explanation",
                requested_difficulty="easy",
                text="Why do deep trees overfit?",
                grounding_quote="Trees overfit.",
                context_ordinals=(1,),
                cited_ordinals=(1,),
                model="qwen3:8b",
                prompt_version="p6-v2",
                params_hash="abc",
            ),
        ],
    )
    connection.commit()


def run_labelling(
    monkeypatch: pytest.MonkeyPatch,
    database_url: str,
    rubric: str,
    keys: list[str],
    limit: int | None = None,
) -> int:
    """Run one labelling pass against a scripted sequence of keypresses."""
    monkeypatch.setenv(DATABASE_URL_ENV, database_url)
    pressed = iter(keys)
    monkeypatch.setattr(cli, "read_key", lambda prompt: next(pressed, "q"))
    argv = ["label-questions", rubric]
    if limit is not None:
        argv += ["--limit", str(limit)]
    return cli.main(argv)


def test_the_display_withholds_everything_the_rubrics_blind(
    connection: Connection, capsys: pytest.CaptureFixture[str]
) -> None:
    """The blinding is the tool's job, not the labeller's discipline."""
    store_two_questions(connection)
    question = next(q for q in list_questions(connection) if q.heading_path == ("S1",))
    cited = chunks_at(connection, [("d1", 0)])

    cli.show_question(question, cited, 1, 2)

    out = capsys.readouterr().out
    assert "Why does bagging reduce variance?" in out
    assert "chunk 0" in out
    assert "Bagging averages." in out
    assert "conceptual" not in out
    assert "hard" not in out
    assert "Bagging averages." not in out.split("CITED MATERIAL")[0]


def test_each_pass_uses_a_different_order(connection: Connection) -> None:
    store_two_questions(connection)
    questions = list_questions(connection)

    orders = {
        name: tuple(
            q.question_id for q in cli.labelling_order(questions, str(spec["seed"]))
        )
        for name, spec in cli.LABEL_PASSES.items()
    }

    assert len(set(orders.values())) > 1


def test_the_order_is_stable_and_independent_of_input_order(
    connection: Connection,
) -> None:
    store_two_questions(connection)
    questions = list_questions(connection)
    seed = str(cli.LABEL_PASSES["groundedness"]["seed"])

    forwards = [q.question_id for q in cli.labelling_order(questions, seed)]
    backwards = [
        q.question_id for q in cli.labelling_order(list(reversed(questions)), seed)
    ]

    assert forwards == backwards


def test_the_pass_seeds_are_what_the_rubrics_declare() -> None:
    assert cli.LABEL_PASSES["groundedness"]["seed"] == "phase6-groundedness-20260907"
    assert cli.LABEL_PASSES["relevance"]["seed"] == "phase6-relevance-20260907"
    assert cli.LABEL_PASSES["difficulty"]["seed"] == "phase6-difficulty-20260907"


def test_a_relevance_pass_records_labels(
    connection: Connection, database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_two_questions(connection)

    assert run_labelling(monkeypatch, database_url, "relevance", ["2", "0"]) == 0

    assert label_totals(connection, "relevance") == {"2": 1, "0": 1}


def test_a_groundedness_failure_prompts_for_its_mode(
    connection: Connection, database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_two_questions(connection)

    run_labelling(monkeypatch, database_url, "groundedness", ["0", "s", "2"])

    assert label_totals(connection, "groundedness") == {"0": 1, "2": 1}
    assert failure_mode_counts(connection) == {"support": 1}


def test_a_supported_grade_is_never_asked_for_a_mode(
    connection: Connection, database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_two_questions(connection)

    run_labelling(monkeypatch, database_url, "groundedness", ["2", "2"])

    assert failure_mode_counts(connection) == {}


def test_a_difficulty_pass_uses_letter_keys(
    connection: Connection, database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_two_questions(connection)

    run_labelling(monkeypatch, database_url, "difficulty", ["e", "u"])

    assert label_totals(connection, "difficulty") == {"easy": 1, "unusable": 1}


def test_quitting_keeps_what_was_already_recorded(
    connection: Connection, database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_two_questions(connection)

    run_labelling(monkeypatch, database_url, "relevance", ["1", "q"])

    assert label_totals(connection, "relevance") == {"1": 1}


def test_a_resumed_pass_offers_only_what_is_left(
    connection: Connection, database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_two_questions(connection)
    run_labelling(monkeypatch, database_url, "relevance", ["1", "q"])

    run_labelling(monkeypatch, database_url, "relevance", ["2"])

    assert label_totals(connection, "relevance") == {"1": 1, "2": 1}


def test_the_session_limit_stops_early(
    connection: Connection, database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_two_questions(connection)

    run_labelling(monkeypatch, database_url, "relevance", ["2", "2"], limit=1)

    assert sum(label_totals(connection, "relevance").values()) == 1


def test_an_unrecognised_key_re_offers_the_same_question(
    connection: Connection, database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_two_questions(connection)

    run_labelling(monkeypatch, database_url, "relevance", ["x", "2", "q"])

    assert label_totals(connection, "relevance") == {"2": 1}


def test_re_judging_backs_out_of_the_failure_mode_prompt(
    connection: Connection, database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_two_questions(connection)

    run_labelling(monkeypatch, database_url, "groundedness", ["1", "r", "2", "q"])

    assert label_totals(connection, "groundedness") == {"2": 1}
    assert failure_mode_counts(connection) == {}

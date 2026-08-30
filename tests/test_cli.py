"""Tests for the command line entry point.

Path expansion and argument parsing need no database. The command tests use the
throwaway database and a stubbed embedder, so no model is contacted.
"""

from __future__ import annotations

from pathlib import Path

import psycopg
import pytest

from daedalus import cli
from daedalus.storage.database import DATABASE_URL_ENV
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

"""Tests for content hashing."""

from __future__ import annotations

import hashlib
from pathlib import Path

from daedalus.core.hashing import file_hash


def test_hash_matches_a_known_digest(tmp_path: Path) -> None:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"attention is all you need")

    assert file_hash(path) == hashlib.sha256(b"attention is all you need").hexdigest()


def test_the_same_bytes_under_different_names_hash_alike(tmp_path: Path) -> None:
    """What makes re-uploading a renamed file idempotent."""

    first = tmp_path / "paper.pdf"
    second = tmp_path / "paper-copy.pdf"
    first.write_bytes(b"identical content")
    second.write_bytes(b"identical content")

    assert file_hash(first) == file_hash(second)


def test_different_bytes_hash_differently(tmp_path: Path) -> None:
    first = tmp_path / "a.pdf"
    second = tmp_path / "b.pdf"
    first.write_bytes(b"one")
    second.write_bytes(b"two")

    assert file_hash(first) != file_hash(second)


def test_an_empty_file_hashes(tmp_path: Path) -> None:
    path = tmp_path / "empty.pdf"
    path.touch()

    assert file_hash(path) == hashlib.sha256(b"").hexdigest()


def test_a_file_larger_than_one_read_block_hashes_correctly(tmp_path: Path) -> None:
    """The incremental read must not drop or reorder blocks."""

    payload = b"x" * (3 * 1024 * 1024 + 7)
    path = tmp_path / "large.pdf"
    path.write_bytes(payload)

    assert file_hash(path) == hashlib.sha256(payload).hexdigest()

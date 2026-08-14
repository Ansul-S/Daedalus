"""
Content hashing.

Ingestion is keyed on what a file contains, not on what it is called. Two
uploads of the same PDF under different names are one document, and a
re-upload after a failed run must be recognised as the same work rather
than indexed twice — the ``content_hash`` unique constraint is what
enforces that, and this is where the value comes from.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["file_hash"]


# Large enough to keep syscall overhead down, small enough that hashing a
# 100 MB PDF does not put the whole file in memory at once.
_READ_SIZE = 1024 * 1024


def file_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes, read incrementally."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while block := handle.read(_READ_SIZE):
            digest.update(block)

    return digest.hexdigest()

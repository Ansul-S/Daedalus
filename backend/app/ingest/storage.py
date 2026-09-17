"""Uploaded files are stored as data/uploads/<sha256><suffix>, so a file is kept only once
however often it is uploaded."""

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

UPLOADS = "uploads"
SOURCE_TYPES = {".pdf": "pdf", ".ipynb": "notebook"}
_READ_SIZE = 1024 * 1024


class UnsupportedFile(ValueError):
    pass


class FileTooLarge(ValueError):
    pass


@dataclass(frozen=True)
class StoredFile:
    # Relative to the data directory
    path: str
    sha256: str
    source_type: str
    size: int


def source_type_for(filename: str) -> str:
    source_type = SOURCE_TYPES.get(Path(filename).suffix.lower())
    if source_type is None:
        raise UnsupportedFile("only PDF (.pdf) and Jupyter notebook (.ipynb) files are supported")
    return source_type


def save_file(data_dir: Path, source: BinaryIO, filename: str, max_bytes: int) -> StoredFile:
    """Copy `source` into the uploads folder, checking its size and type on the way."""
    source_type = source_type_for(filename)
    uploads = data_dir / UPLOADS
    uploads.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with tempfile.NamedTemporaryFile(dir=uploads, suffix=".part", delete=False) as temporary:
        try:
            while block := source.read(_READ_SIZE):
                size += len(block)
                if size > max_bytes:
                    raise FileTooLarge(f"file is larger than {max_bytes // (1024 * 1024)} MB")
                digest.update(block)
                temporary.write(block)
            temporary.close()
            _check_content(Path(temporary.name), source_type)
            name = f"{digest.hexdigest()}{Path(filename).suffix.lower()}"
            os.replace(temporary.name, uploads / name)
        except BaseException:
            Path(temporary.name).unlink(missing_ok=True)
            raise
    return StoredFile(f"{UPLOADS}/{name}", digest.hexdigest(), source_type, size)


def _check_content(path: Path, source_type: str) -> None:
    if source_type == "pdf":
        with path.open("rb") as file:
            if not file.read(5).startswith(b"%PDF-"):
                raise UnsupportedFile("the file is not a PDF")
        return
    try:
        notebook = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UnsupportedFile("the file is not a Jupyter notebook (invalid JSON)") from exc
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise UnsupportedFile("the file is not a Jupyter notebook (no cells)")

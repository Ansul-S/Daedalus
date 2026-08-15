"""
Freezing the parsed corpus.

Every evaluation label is a character range into a document's parsed text.
That only means something if the text never moves — and it moves for
reasons outside anyone's control: a vision model describes an image
differently on each run (ADR-009), and a parser upgrade reflows a PDF. Any
of those silently invalidates every label downstream of the change.

So extraction happens once, the output is committed, and the harness reads
the committed files. Re-parsing becomes a deliberate migration with a
version bump and a re-anchoring pass, rather than something that happens
by accident on a Tuesday.

The frozen form keeps segments as well as text. Without them the
extraction method and page of every chunk would be lost, and metrics could
not be sliced by ingestion path — which is most of why the ``extraction``
column exists at all.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from daedalus.config import constants
from daedalus.ingestion.router import make_doc_id, parse
from daedalus.ingestion.types import ParsedDocument, Segment

__all__ = [
    "FrozenDocument",
    "freeze_corpus",
    "load_frozen",
    "load_manifest",
    "text_path_for",
]


logger = logging.getLogger(__name__)


# Bumped when a change to parsing invalidates existing labels. Stored in the
# manifest so a dataset can state which version it was labelled against.
FROZEN_VERSION = 1


@dataclass(frozen=True)
class FrozenDocument:
    """A document's parsed text and everything needed to re-chunk it."""

    doc_id: str
    filename: str
    source_type: str
    text: str
    segments: tuple[Segment, ...]
    n_pages: int | None
    text_sha256: str

    def as_parsed(self) -> ParsedDocument:
        """Rebuild the parser's output without running the parser."""

        return ParsedDocument(
            doc_id=self.doc_id,
            text=self.text,
            segments=self.segments,
            source_type=self.source_type,
            n_pages=self.n_pages,
        )


def text_path_for(doc_id: str) -> Path:
    return constants.EVAL_PARSED_DIR / f"{doc_id}.txt"


# Newline translation is disabled on both sides. Python's text mode rewrites
# \r and \r\n to \n on read, and notebook outputs are full of carriage
# returns from progress bars — so a round trip through the default reader
# silently shortens the text and shifts every offset after the first one.
def _write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def _read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_type_for(path: Path) -> str:
    """
    Classify by the corpus directory a document sits in.

    Richer than the extension-based classification the upload endpoint
    uses: ``arxiv`` and ``course_notes`` are both PDFs, and
    EVALUATION_ENGINE.md slices results by that distinction.
    """

    folder = path.parent.name.lower()

    if folder.startswith("arxiv"):
        return "arxiv"

    if folder.startswith("course"):
        return "course_notes"

    if folder.startswith("notebook"):
        return "notebook"

    if folder.startswith("image"):
        return "image"

    return "unknown"


def freeze_corpus(
    corpus_dir: Path | None = None, destination: Path | None = None
) -> list[FrozenDocument]:
    """
    Parse every supported document once and write the result to disk.

    Documents that cannot be parsed are logged and skipped rather than
    failing the run: the image path needs a vision model that CI will not
    have, and a corpus that is 90% frozen is still a usable benchmark.
    """

    source = corpus_dir or constants.CORPUS_DIR
    target = destination or constants.EVAL_PARSED_DIR

    target.mkdir(parents=True, exist_ok=True)

    frozen: list[FrozenDocument] = []

    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in constants.SUPPORTED_EXTENSIONS:
            continue

        doc_id = make_doc_id(path)
        source_type = _source_type_for(path)

        try:
            parsed = parse(path, source_type=source_type, doc_id=doc_id)
        except Exception as error:
            logger.warning("Skipping %s: %s", path.name, error)
            continue

        document = FrozenDocument(
            doc_id=doc_id,
            filename=path.name,
            source_type=source_type,
            text=parsed.text,
            segments=parsed.segments,
            n_pages=parsed.n_pages,
            text_sha256=_digest(parsed.text),
        )

        _write_text(text_path_for(doc_id), document.text)
        frozen.append(document)

        logger.info("Froze %s: %d chars", doc_id, len(document.text))

    _write_manifest(frozen)

    return frozen


def _write_manifest(documents: list[FrozenDocument]) -> None:
    """Record what was frozen, so a mismatch is detectable rather than silent."""

    manifest = {
        "version": FROZEN_VERSION,
        "documents": [
            {
                "doc_id": document.doc_id,
                "filename": document.filename,
                "source_type": document.source_type,
                "n_pages": document.n_pages,
                "n_chars": len(document.text),
                "text_sha256": document.text_sha256,
                "segments": [
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "extraction": segment.extraction,
                        "page": segment.page,
                    }
                    for segment in document.segments
                ],
            }
            for document in documents
        ],
    }

    constants.EVAL_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_manifest(path: Path | None = None) -> dict[str, object]:
    """Read the manifest describing the frozen corpus."""

    target = path or constants.EVAL_MANIFEST_PATH

    if not target.exists():
        raise FileNotFoundError(
            f"no frozen corpus at {target}. Run: uv run python -m daedalus.evaluation.freeze"
        )

    loaded: dict[str, object] = json.loads(target.read_text(encoding="utf-8"))

    return loaded


def load_frozen(manifest_path: Path | None = None) -> list[FrozenDocument]:
    """
    Load every frozen document, verifying it has not drifted.

    The hash check is the point of the manifest: an edited or re-parsed
    text file would shift offsets and quietly invalidate every label
    anchored to it, and a wrong number is worse than a missing one.
    """

    manifest = load_manifest(manifest_path)
    entries = manifest["documents"]

    if not isinstance(entries, list):  # pragma: no cover - malformed manifest
        raise ValueError("manifest is malformed: 'documents' is not a list")

    documents: list[FrozenDocument] = []

    for entry in entries:
        doc_id = str(entry["doc_id"])
        text = _read_text(text_path_for(doc_id))
        recorded = str(entry["text_sha256"])

        if _digest(text) != recorded:
            raise ValueError(
                f"{doc_id}: frozen text does not match the manifest hash. "
                f"Every label anchored to it is now suspect — re-freeze and re-anchor."
            )

        documents.append(
            FrozenDocument(
                doc_id=doc_id,
                filename=str(entry["filename"]),
                source_type=str(entry["source_type"]),
                text=text,
                segments=tuple(
                    Segment(
                        start=segment["start"],
                        end=segment["end"],
                        extraction=segment["extraction"],
                        page=segment["page"],
                    )
                    for segment in entry["segments"]
                ),
                n_pages=entry["n_pages"],
                text_sha256=recorded,
            )
        )

    return documents

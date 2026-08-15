"""
Freeze the corpus.

    uv run python -m daedalus.evaluation.freeze

Run deliberately, not automatically. Overwriting the frozen text shifts the
character offsets every label is anchored to, so this is a migration: bump
``FROZEN_VERSION``, re-run, and re-anchor the datasets.
"""

from __future__ import annotations

from daedalus.config import constants
from daedalus.core.logging import configure_logging
from daedalus.evaluation.corpus import freeze_corpus


def main() -> None:
    configure_logging()

    documents = freeze_corpus()

    total = sum(len(document.text) for document in documents)

    print(f"\nFroze {len(documents)} documents, {total:,} characters")
    print(f"  text     -> {constants.EVAL_PARSED_DIR}")
    print(f"  manifest -> {constants.EVAL_MANIFEST_PATH}\n")

    for document in sorted(documents, key=lambda item: item.source_type):
        print(
            f"  {document.source_type:<13} {document.doc_id:<34} "
            f"{len(document.text):>8,} chars  {len(document.segments):>3} segments"
        )


if __name__ == "__main__":
    main()

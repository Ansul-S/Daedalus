"""Generate one interview question for each section of the frozen selection.

The protocol is `docs/PHASE-6-PROTOCOL.md`, committed before this was run. Every
parameter it fixes lives as a constant under `daedalus.generation` and none is
exposed as a flag here: the eligibility threshold, the selection seed, the draw
size, the type mapping, the difficulty rotation, the context budget, the model,
the temperature and the per-section decoding seed are all frozen. A flag is an
invitation to try another value and report the better one.

The two options that do exist are operational rather than experimental.
`--limit` caps how many sections the model is called for, so a small run can
verify the pipeline before the full draw is committed to. `--dry-run` builds
every prompt and writes the report without contacting the model at all.

Rerunning is safe. A section that already holds a question, or already holds a
rejection, is skipped rather than generated again.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from daedalus.generation.context import NONPROSE_BUDGET_CHARS, section_context
from daedalus.generation.prompt import (
    GENERATION_MODEL,
    NUM_PREDICT,
    PROMPT_VERSION,
    TEMPERATURE,
    build_messages,
    generation_options,
    params_hash,
)
from daedalus.generation.runner import RunReport, SectionOutcome, run
from daedalus.generation.selection import (
    MIN_PROSE_CHARS,
    SELECTION_SEED,
    SELECTION_SIZE,
    SelectedSection,
    difficulty_counts,
    load_sections,
    partition,
    select,
    type_counts,
)
from daedalus.storage.database import connect
from daedalus.storage.questions import rejection_counts

#: Where the run report is written. `results/` is tracked; this is an artifact.
DEFAULT_OUT = Path("results")


def outcome_row(outcome: SectionOutcome) -> dict[str, object]:
    """One section's result, as it appears in the artifact."""
    return {
        "selection_rank": outcome.selection_rank,
        "doc_id": outcome.doc_id,
        "heading_path": list(outcome.heading_path),
        "outcome": outcome.outcome,
        "reason": outcome.reason,
        "detail": outcome.detail,
        "question_id": outcome.question_id,
    }


def build_artifact(
    report: RunReport,
    selected: Sequence[SelectedSection],
    eligible: int,
    thin: int,
    prose_free: int,
    dry_run: bool,
) -> dict[str, object]:
    """Assemble the run artifact, including the frozen parameters it ran under."""
    total = eligible + thin + prose_free
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "protocol": "docs/PHASE-6-PROTOCOL.md",
        "dry_run": dry_run,
        "parameters": {
            "min_prose_chars": MIN_PROSE_CHARS,
            "nonprose_budget_chars": NONPROSE_BUDGET_CHARS,
            "selection_seed": SELECTION_SEED,
            "selection_size": SELECTION_SIZE,
            "model": GENERATION_MODEL,
            "prompt_version": PROMPT_VERSION,
            "params_hash": params_hash(),
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT,
        },
        "sections": {
            "total": total,
            "eligible": eligible,
            "thin": thin,
            "prose_free": prose_free,
            "selected": len(selected),
            "selected_of_eligible": round(len(selected) / eligible, 4)
            if eligible
            else None,
            "eligible_of_total": round(eligible / total, 4) if total else None,
            "selected_of_total": round(len(selected) / total, 4) if total else None,
        },
        "requested": {
            "types": dict(sorted(type_counts(selected).items())),
            "difficulty": dict(sorted(difficulty_counts(selected).items())),
        },
        "run": {
            "attempted": report.attempted,
            "accepted": report.accepted,
            "rejected": report.rejected,
            "failed": report.failed,
            "skipped": report.skipped,
        },
        "rejections_by_reason": dict(sorted(report.rejection_counts().items())),
        "failures_by_reason": dict(sorted(report.failure_counts().items())),
        "outcomes": [outcome_row(o) for o in report.outcomes],
    }


def report_lines(artifact: dict[str, object]) -> list[str]:
    """The summary printed to the terminal."""
    sections = artifact["sections"]
    run_counts = artifact["run"]
    assert isinstance(sections, dict) and isinstance(run_counts, dict)

    lines = [
        "=" * 70,
        "GENERATION RUN"
        + ("  (DRY RUN - no model contacted)" if artifact["dry_run"] else ""),
        "=" * 70,
        "  protocol        docs/PHASE-6-PROTOCOL.md",
        f"  model           {GENERATION_MODEL}   prompt {PROMPT_VERSION}   "
        f"params {params_hash()[:8]}",
        "",
        f"  sections        {sections['total']} total",
        f"    eligible      {sections['eligible']}  "
        f"(thin {sections['thin']}, prose-free {sections['prose_free']})",
        f"    selected      {sections['selected']}",
        f"    coverage      {sections['selected']}/{sections['eligible']} selected of "
        f"eligible, {sections['eligible']}/{sections['total']} eligible of all, "
        f"{sections['selected']}/{sections['total']} generated of all",
        "",
        f"  attempted       {run_counts['attempted']}",
        f"    accepted      {run_counts['accepted']}",
        f"    rejected      {run_counts['rejected']}",
        f"    failed        {run_counts['failed']}   (transport, retryable)",
        f"    skipped       {run_counts['skipped']}  (already attempted)",
    ]

    rejections = artifact["rejections_by_reason"]
    failures = artifact["failures_by_reason"]
    assert isinstance(rejections, dict) and isinstance(failures, dict)

    if rejections:
        lines.extend(["", "  rejections by reason"])
        lines.extend(f"    {reason:22} {count}" for reason, count in rejections.items())
    if failures:
        lines.extend(["", "  transport failures by reason (NOT rejections)"])
        lines.extend(f"    {reason:22} {count}" for reason, count in failures.items())

    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap how many sections the model is called for",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build every prompt and write the report without contacting the model",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    with connect() as connection:
        classes = partition(load_sections(connection))
        selected = select(classes.eligible)

        if args.dry_run:
            for item in selected[: args.limit] if args.limit else selected:
                context = section_context(
                    connection, item.section.doc_id, item.section.heading_path
                )
                build_messages(context, item.question_type, item.difficulty)
                generation_options(item.section)
            report = RunReport(
                model=GENERATION_MODEL,
                prompt_version=PROMPT_VERSION,
                params_hash=params_hash(),
            )
        else:

            def progress(outcome: SectionOutcome) -> None:
                """Report a section and commit it before moving on.

                Committing per section is what makes the skip-on-rerun design
                usable. Holding the whole walk in one transaction would discard
                every completed section if the process died partway, and the
                rerun would then have nothing to skip.
                """
                print(
                    f"  rank {outcome.selection_rank:>3}  {outcome.outcome:9}"
                    f"{'  ' + outcome.reason if outcome.reason else ''}",
                    flush=True,
                )
                connection.commit()

            report = run(connection, selected, limit=args.limit, on_progress=progress)

        artifact = build_artifact(
            report,
            selected,
            len(classes.eligible),
            len(classes.thin),
            len(classes.prose_free),
            args.dry_run,
        )

        if not args.dry_run:
            persisted = rejection_counts(connection)
            artifact["rejections_in_store"] = dict(sorted(persisted.items()))

    print()
    print("\n".join(report_lines(artifact)))

    out = args.out or DEFAULT_OUT / (
        f"questions_{'dryrun_' if args.dry_run else ''}"
        f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

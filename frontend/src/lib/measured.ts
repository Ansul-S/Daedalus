import { readFileSync } from "node:fs";
import path from "node:path";

// The grader's measurement, read from the design notes when the landing page is built, so the
// page never states a number the notes don't. The build fails if the notes stop saying it.

const DESIGN_NOTES = path.join(process.cwd(), "..", "docs", "design.md");
// The milestone table's row for grading
const MILESTONE = /\*\*Passed\*\*: ρ ([\d.]+) and Cohen's κ ([\d.]+) on (\d+) hand-graded answers/;
// The section that tells how it was measured, and its address on the page
const METHOD = "## Phase 3 milestone result";

/** Where the design notes are read, on GitHub. */
export const DESIGN_NOTES_URL = "https://github.com/Ansul-S/Daedalus/blob/main/docs/design.md";

export type Measurement = {
  spearman: string;
  kappa: string;
  answers: string;
  /** The design notes' section on how it was measured. */
  method: string;
};

export function graderMeasurement(): Measurement {
  const notes = readFileSync(DESIGN_NOTES, "utf-8");
  const row = MILESTONE.exec(notes);
  if (!row || !notes.includes(`\n${METHOD}\n`)) {
    throw new Error(
      `${DESIGN_NOTES} no longer has the grader's milestone row ("**Passed**: ρ … and Cohen's κ … on … hand-graded answers") or its "${METHOD}" section, which the landing page quotes. Change src/lib/measured.ts to read them from where they are now.`,
    );
  }
  const anchor = METHOD.replace(/^#+ /, "").toLowerCase().replaceAll(" ", "-");
  return {
    spearman: row[1],
    kappa: row[2],
    answers: row[3],
    method: `${DESIGN_NOTES_URL}#${anchor}`,
  };
}

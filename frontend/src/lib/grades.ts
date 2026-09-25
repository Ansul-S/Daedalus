import type { GradeOut, KeyPointGradeOut, ReviewOut } from "@/client/types.gen";
import type { KeyPointState, Rating, Verdict } from "@/components/hatching";
import { dayLabel } from "@/lib/time";

// How a grade is read out on the practice page. The score itself comes from the API; the
// arithmetic here only shows how it was reached (backend/app/grading/scoring.py).

/** What the API takes (backend/app/api/grading.py). */
export const MAX_ANSWER_CHARS = 8000;
export const MAX_SECONDS = 24 * 60 * 60;

const CREDIT: Record<KeyPointState, number> = { covered: 1, partial: 0.5, missing: 0 };
const CONTRADICTION_PENALTY = 0.15;

export const RATING_LABEL: Record<Rating, string> = {
  again: "Again",
  hard: "Hard",
  good: "Good",
  easy: "Easy",
};

export function keyPointState(status: string): KeyPointState {
  return status === "covered" || status === "partial" ? status : "missing";
}

export function verdictOf(verdict: string): Verdict {
  return verdict === "supported" || verdict === "contradicted" ? verdict : "unverified";
}

function amount(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

/** coverage 6 / 9 = 0.67 · − 0.15 × 1 contradicted claim · = 0.52 */
export function scoreFormula(grade: GradeOut): string | null {
  if (grade.score === null || grade.coverage === null) return null;
  const total = grade.key_points.reduce((sum, point) => sum + point.weight, 0);
  const earned = grade.key_points.reduce(
    (sum, point) => sum + point.weight * CREDIT[keyPointState(point.status)],
    0,
  );
  const contradicted = grade.contradicted ?? 0;
  const floored = grade.coverage - CONTRADICTION_PENALTY * contradicted < 0;
  return [
    `coverage ${amount(earned)} / ${total} = ${grade.coverage.toFixed(2)}`,
    contradicted === 0
      ? "no claim contradicted"
      : `− ${CONTRADICTION_PENALTY} × ${contradicted} contradicted claim${contradicted === 1 ? "" : "s"}`,
    `= ${grade.score.toFixed(2)}${floored ? " (never below 0)" : ""}`,
  ].join(" · ");
}

/** See again in 2 days · Fri 2 Oct */
export function comeBack(review: ReviewOut): string {
  const when =
    review.interval <= 0
      ? "today"
      : review.interval === 1
        ? "tomorrow"
        : `in ${review.interval} days`;
  return `See again ${when} · ${dayLabel(review.due)}`;
}

/** Graded by qwen/qwen3.8-27b in 3.5 s · 2,232 tokens */
export function gradedBy(grade: GradeOut): string | null {
  const count = (key: string) => {
    const value = grade.usage[key];
    return typeof value === "number" ? value : 0;
  };
  const tokens = count("input_tokens") + count("output_tokens");
  const parts = [
    grade.grader_model && `Graded by ${grade.grader_model}`,
    grade.seconds !== null && `in ${grade.seconds.toFixed(1)} s`,
  ].filter(Boolean);
  if (parts.length === 0) return null;
  return tokens > 0 ? `${parts.join(" ")} · ${tokens.toLocaleString("en")} tokens` : parts.join(" ");
}

// ---------- the answer's own words, found again in the answer ----------

export type Highlight = { start: number; end: number; point: KeyPointGradeOut };

// The folding the grader's quote check does (backend/app/questions/grounding.py), kept to
// changes that map one character to one, so a match can be traced back to the answer.
const HYPHENS = /[‐-—−]/;
const MARKS = /[$`*_#\s]/;
const ELLIPSIS = /\.\.\.|…/;
const TRAILING = /[\s.,:;!?-]+$/;
// Shorter runs than this would be found anywhere: a quote that short is shown, not traced.
const MIN_WORDS = 3;

/** The text folded for matching, with each folded character's place in the original. */
function fold(text: string): { folded: string; at: number[] } {
  let folded = "";
  const at: number[] = [];
  let gap = true;
  for (let i = 0; i < text.length; i++) {
    let char = text[i];
    if (MARKS.test(char)) {
      if (!gap) {
        folded += " ";
        at.push(i);
        gap = true;
      }
      continue;
    }
    if (HYPHENS.test(char)) char = "-";
    const lower = char.toLowerCase();
    folded += lower.length === 1 ? lower : char;
    at.push(i);
    gap = false;
  }
  return { folded, at };
}

/** A quote's stretches of words: an ellipsis stands for words the grader left out. */
function stretches(quote: string): string[] {
  return quote
    .split(ELLIPSIS)
    .map((part) => fold(part).folded.trim().replace(TRAILING, ""))
    .filter((part) => part.split(" ").length >= MIN_WORDS);
}

/** Where each covered or partial key point's quote sits in the answer. Quotes the grader
 * paraphrased (quote_found false) or that can't be placed exactly are left untraced. */
export function traceQuotes(answer: string, points: KeyPointGradeOut[]): Highlight[] {
  const { folded, at } = fold(answer);
  const found: Highlight[] = [];
  for (const point of points) {
    if (keyPointState(point.status) === "missing" || point.quote_found !== true) continue;
    let from = 0;
    for (const stretch of stretches(point.answer_quote)) {
      const index = folded.indexOf(stretch, from);
      if (index < 0) break;
      found.push({ start: at[index], end: at[index + stretch.length - 1] + 1, point });
      from = index + stretch.length;
    }
  }
  // Two points may quote the same words: the first one keeps them.
  found.sort((a, b) => a.start - b.start || b.end - a.end);
  const kept: Highlight[] = [];
  for (const highlight of found) {
    if (kept.length === 0 || highlight.start >= kept[kept.length - 1].end) kept.push(highlight);
  }
  return kept;
}

/** The answer cut into plain runs and traced ones, in order. */
export function splitAnswer(
  answer: string,
  highlights: Highlight[],
): { text: string; point: KeyPointGradeOut | null }[] {
  const pieces: { text: string; point: KeyPointGradeOut | null }[] = [];
  let cursor = 0;
  for (const { start, end, point } of highlights) {
    if (start > cursor) pieces.push({ text: answer.slice(cursor, start), point: null });
    pieces.push({ text: answer.slice(start, end), point });
    cursor = end;
  }
  if (cursor < answer.length) pieces.push({ text: answer.slice(cursor), point: null });
  return pieces;
}

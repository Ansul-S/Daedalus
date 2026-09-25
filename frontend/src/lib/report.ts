// A question's validation report, read for its page. The report is JSON kept with the
// question: what each check found when it was written (backend/app/questions/validation.py),
// then every correction made to it since (backend/app/api/questions.py). It is read
// defensively, since nothing in its type says which keys are there.

export type QuoteCheck = {
  quote: string;
  chunkId: number | null;
  /** How closely the quote matched its passage, 0 to 100 */
  score: number | null;
  /** Why it failed; null for a quote that was found */
  problem: string | null;
};

export type Change = { field: string; from: unknown; to: unknown };

export type Edit = {
  at: string;
  reason: string | null;
  changes: Change[];
  /** The new key points' quotes, each checked against its passage */
  quotes: QuoteCheck[];
  /** When the text changed: whether its embedding was made again or cleared */
  embedding: string | null;
};

export type Report = {
  /** The checks that turned the question down, in the order they ran */
  failed: string[];
  quotes: QuoteCheck[];
  /** The checker, a second model reading only the passages */
  answerable: boolean | null;
  missing: string | null;
  /** "explain", or "recall" for trivia */
  kind: string | null;
  checkerModel: string | null;
  promptVersion: string | null;
  checkerAnswer: string | null;
  /** How close the reference answer is to the checker's, 0 to 1; recorded, not judged */
  agreement: number | null;
  /** Whether the duplicate check ran: it needs the local embedding model */
  duplicateChecked: boolean;
  nearest: { id: number; similarity: number | null } | null;
  edits: Edit[];
};

// The four checks, in the order they run
export const CHECKS = ["quotes", "answerable", "trivia", "duplicate"] as const;
export type Check = (typeof CHECKS)[number];

type Json = Record<string, unknown>;

function record(value: unknown): Json | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Json)
    : null;
}

function text(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function figure(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function records(value: unknown): Json[] {
  return Array.isArray(value) ? value.map(record).filter((item) => item !== null) : [];
}

function quotes(value: unknown): QuoteCheck[] {
  return records(value).map((quote) => ({
    quote: text(quote.quote) ?? "",
    chunkId: figure(quote.chunk_id),
    score: figure(quote.score),
    problem: text(quote.problem),
  }));
}

// The order a question's page reads in; the database keeps a report's keys in its own order
const FIELD_ORDER = ["text", "reference_answer", "key_points", "status"];

function rank(field: string): number {
  const found = FIELD_ORDER.indexOf(field);
  return found === -1 ? FIELD_ORDER.length : found;
}

function edits(value: unknown): Edit[] {
  return records(value).map((edit) => ({
    at: text(edit.at) ?? "",
    reason: text(edit.reason),
    changes: Object.entries(record(edit.changes) ?? {})
      .map(([field, change]) => ({ field, from: record(change)?.from, to: record(change)?.to }))
      .sort((a, b) => rank(a.field) - rank(b.field)),
    quotes: quotes(edit.quotes),
    embedding: text(edit.embedding),
  }));
}

export function readReport(validation: Json): Report {
  const nearest = figure(validation.nearest_question);
  return {
    failed: Array.isArray(validation.failed)
      ? validation.failed.filter((name) => typeof name === "string")
      : [],
    quotes: quotes(validation.quotes),
    answerable: typeof validation.answerable === "boolean" ? validation.answerable : null,
    missing: text(validation.missing),
    kind: text(validation.kind),
    checkerModel: text(validation.checker_model),
    promptVersion: text(validation.prompt_version),
    checkerAnswer: text(validation.checker_answer),
    agreement: figure(validation.answer_agreement),
    duplicateChecked: validation.duplicate !== "not checked",
    nearest: nearest === null ? null : { id: nearest, similarity: figure(validation.nearest_similarity) },
    edits: edits(validation.edits),
  };
}

/** Whether the checker named something the passages lack ("nothing" when they lack nothing). */
export function missingSomething(missing: string | null): missing is string {
  return missing !== null && missing.trim() !== "" && missing.trim().toLowerCase() !== "nothing";
}

const FIELD_NAMES: Record<string, string> = {
  text: "the question",
  reference_answer: "the reference answer",
  key_points: "the key points",
  status: "the status",
};

/** The question, the reference answer and the key points: what an edit changed, in words. */
export function changedFields(changes: Change[]): string {
  const names = changes.map((change) => FIELD_NAMES[change.field] ?? change.field);
  if (names.length <= 1) return names.join("");
  return `${names.slice(0, -1).join(", ")} and ${names.at(-1)}`;
}

/** A key point as a correction recorded it. */
export type RecordedPoint = { text: string; weight: number | null };

export function recordedPoints(value: unknown): RecordedPoint[] {
  return records(value).map((point) => ({
    text: text(point.text) ?? "",
    weight: figure(point.weight),
  }));
}

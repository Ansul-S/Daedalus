// How a question is named on screen, and what the API takes when one is corrected.

const STYLES: Record<string, string> = {
  intuition: "intuition",
  why_how: "why / how",
  compare: "compare",
  tradeoffs: "trade-offs",
  failure_modes: "failure modes",
  connection: "connection",
  paper: "paper",
};

/** The seven shapes a question can take, in the order the design notes list them. */
export const STYLE_NAMES = Object.keys(STYLES);

export function styleLabel(style: string): string {
  return STYLES[style] ?? style.replaceAll("_", " ");
}

/** Q.043 */
export function questionNumber(id: number): string {
  return `Q.${String(id).padStart(3, "0")}`;
}

/** Where a question's page is. */
export function questionPath(id: number): string {
  return `/questions/${id}`;
}

/** Accepted questions are the library practice draws from; the checks turned the rejected
 * ones down, and a retired one was taken out by hand. */
export const STATUSES = ["accepted", "retired", "rejected"] as const;
export type Status = (typeof STATUSES)[number];

export const STATUS_LABEL: Record<Status, string> = {
  accepted: "Accepted",
  retired: "Retired",
  rejected: "Rejected",
};

export function statusLabel(status: string): string {
  return STATUS_LABEL[status as Status] ?? status;
}

/** What a correction may hold (backend/app/api/questions.py). */
export const MAX_TEXT_CHARS = 1000;
export const MAX_REFERENCE_CHARS = 3000;
export const MAX_POINT_CHARS = 500;
export const MAX_QUOTE_CHARS = 1000;
export const MAX_REASON_CHARS = 500;
export const MIN_KEY_POINTS = 2;
export const MAX_KEY_POINTS = 4;

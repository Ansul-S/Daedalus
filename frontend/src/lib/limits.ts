// The daily limits on grading (backend/app/grading/limits.py), as the practice page tells them.

import type { LimitOut } from "@/client/types.gen";
import { ApiError } from "@/lib/api-errors";

/** The daily limit a grade was refused by, when that is why the API refused it (429). */
export function refusalOf(error: unknown): LimitOut | null {
  if (!(error instanceof ApiError) || error.status !== 429) return null;
  const { body } = error;
  if (typeof body !== "object" || body === null || !("limit" in body)) return null;
  return (body as { limit: LimitOut }).limit;
}

function hhmm(at: Date): string {
  return `${String(at.getHours()).padStart(2, "0")}:${String(at.getMinutes()).padStart(2, "0")}`;
}

function sameDay(one: Date, other: Date): boolean {
  return one.toDateString() === other.toDateString();
}

/** When a grade is allowed again, in the reader's own time: "at 14:32", "tomorrow at 04:00",
 * or "on 2 Oct at 04:00". */
export function againLabel(iso: string, now: Date = new Date()): string {
  const at = new Date(iso);
  const tomorrow = new Date(now);
  tomorrow.setDate(now.getDate() + 1);
  if (sameDay(at, now)) return `at ${hhmm(at)}`;
  if (sameDay(at, tomorrow)) return `tomorrow at ${hhmm(at)}`;
  const day = at.toLocaleDateString("en-GB", { day: "numeric", month: "short" });
  return `on ${day} at ${hhmm(at)}`;
}

/** Why no grade can be asked for now, in a sentence. */
export function limitReason(limit: LimitOut): string {
  if (limit.again_at === null) return "Grading is switched off for now.";
  const again = againLabel(limit.again_at);
  if (limit.scope === "per_user") {
    return `You have used today's ${limit.allowed} grades; more ${again}.`;
  }
  const all = `Everyone together has used the ${limit.allowed} grades a day allows`;
  return `${all}; the next is free ${again}.`;
}

/** The same, in a few words beside the answer. */
export function limitShort(limit: LimitOut): string {
  return limit.again_at === null
    ? "Grading is off for now"
    : `No grades left · more ${againLabel(limit.again_at)}`;
}

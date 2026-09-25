// An answer being written is kept in this browser until it is graded, so a reload or a closed
// tab loses nothing. It never leaves the machine before it is submitted.

const PREFIX = "daedalus:draft:";

export type Draft = { answer: string; seconds: number };

export function loadDraft(questionId: number): Draft | null {
  try {
    const saved = JSON.parse(localStorage.getItem(PREFIX + questionId) ?? "null") as unknown;
    if (typeof saved !== "object" || saved === null) return null;
    const { answer, seconds } = saved as Partial<Draft>;
    if (typeof answer !== "string") return null;
    return { answer, seconds: typeof seconds === "number" && seconds >= 0 ? seconds : 0 };
  } catch {
    return null; // storage unavailable, or not ours to read
  }
}

export function saveDraft(questionId: number, draft: Draft): void {
  try {
    if (draft.answer.trim()) localStorage.setItem(PREFIX + questionId, JSON.stringify(draft));
    else localStorage.removeItem(PREFIX + questionId);
  } catch {
    // storage unavailable (private mode, blocked): the draft lasts as long as the page
  }
}

export function clearDraft(questionId: number): void {
  try {
    localStorage.removeItem(PREFIX + questionId);
  } catch {
    // nothing was kept
  }
}

import { client } from "@/client/client.gen";

/** A failed API call: its HTTP status, or null when the API could not be reached at all. */
export class ApiError extends Error {
  constructor(
    readonly status: number | null,
    readonly body: unknown,
  ) {
    super(
      status === null
        ? "The API could not be reached"
        : (detailOf(body) ?? `The API answered ${status}`),
    );
    this.name = "ApiError";
  }
}

/** FastAPI's error detail: its message, or the first validation error's. */
export function detailOf(body: unknown): string | null {
  if (typeof body === "string") return body || null;
  if (typeof body !== "object" || body === null || !("detail" in body)) return null;
  const { detail } = body as { detail: unknown };
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && typeof detail[0]?.msg === "string") return plain(detail[0].msg);
  return null;
}

// Pydantic opens a validator's own message with the kind of error
function plain(message: string): string {
  return message.replace(/^Value error, /, "");
}

/** A refused request's problems, keyed by the field each points at, as FastAPI's `loc` gives
 * it inside the body: "text", "key_points.1.evidence_quote", or "" for the request as a
 * whole. The first problem of each field is kept; any other error gives none. */
export function fieldErrors(error: unknown): Map<string, string> {
  const found = new Map<string, string>();
  if (!(error instanceof ApiError) || error.status !== 422) return found;
  const body = error.body;
  const detail = typeof body === "object" && body !== null && "detail" in body ? body.detail : null;
  if (!Array.isArray(detail)) return found;
  for (const problem of detail) {
    if (typeof problem !== "object" || problem === null) continue;
    const { loc, msg } = problem as { loc?: unknown; msg?: unknown };
    if (!Array.isArray(loc) || typeof msg !== "string") continue;
    const field = (loc[0] === "body" ? loc.slice(1) : loc).join(".");
    if (!found.has(field)) found.set(field, plain(msg));
  }
  return found;
}

// Every generated call and query now fails with an ApiError, so a page can tell "not found"
// from an API that isn't running. A cancelled request stays a cancellation.
client.interceptors.error.use((error, response) => {
  if (error instanceof ApiError) return error;
  if (error instanceof DOMException && error.name === "AbortError") return error;
  return new ApiError(response?.status ?? null, error);
});

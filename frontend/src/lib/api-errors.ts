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
  if (Array.isArray(detail) && typeof detail[0]?.msg === "string") return detail[0].msg;
  return null;
}

// Every generated call and query now fails with an ApiError, so a page can tell "not found"
// from an API that isn't running. A cancelled request stays a cancellation.
client.interceptors.error.use((error, response) => {
  if (error instanceof ApiError) return error;
  if (error instanceof DOMException && error.name === "AbortError") return error;
  return new ApiError(response?.status ?? null, error);
});

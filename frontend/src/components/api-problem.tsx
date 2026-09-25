import { Button } from "@/components/ui/button";
import { API_URL } from "@/lib/api";
import { ApiError } from "@/lib/api-errors";

/** A page's data didn't come: the API isn't running, or it answered with an error. The error
 * is an ApiError whatever the generated client's types say (src/lib/api-errors.ts). */
export function ApiProblem({ error, retry }: { error: unknown; retry: () => void }) {
  const status = error instanceof ApiError ? error.status : null;
  return (
    <div className="max-w-[62ch]">
      <p className="type-label text-fg-2">
        {status === null ? "API not reachable" : `API error ${status}`}
      </p>
      <p className="mt-3 mb-4">
        {status === null ? (
          <>
            Can&apos;t reach the API at <code className="font-mono">{API_URL}</code>. Start it
            with <code className="font-mono">make api</code>, then try again.
          </>
        ) : error instanceof Error ? (
          error.message
        ) : (
          "The API refused the request."
        )}
      </p>
      <Button variant="outline" size="sm" onClick={retry}>
        Try again
      </Button>
    </div>
  );
}

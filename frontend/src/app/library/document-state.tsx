import type { DocumentOut } from "@/client/types.gen";
import { JobLine } from "@/components/job-status";
import { isActive, jobDoing, jobState, plural } from "@/lib/library";

/** Where a document stands: its latest reading while it waits or runs, then ready or failed.
 * A document read before keeps its passages when a later reading fails. `ready` says more
 * about a ready document: how many passages it made, unless told otherwise. */
export function DocumentState({
  document,
  worker,
  ready,
}: {
  document: DocumentOut;
  worker: boolean | undefined;
  ready?: React.ReactNode;
}) {
  const job = document.latest_job;
  if (isActive(job)) {
    const state = jobState(job, worker);
    return <JobLine state={state}>{jobDoing(job, state)}</JobLine>;
  }
  if (document.status === "ready") {
    return (
      <JobLine state="done" label="Ready">
        {ready ?? plural(document.chunk_count ?? 0, "passage")}
      </JobLine>
    );
  }
  if (document.status === "failed") return <JobLine state="failed" />;
  return <JobLine state="queued" label="Pending" />;
}

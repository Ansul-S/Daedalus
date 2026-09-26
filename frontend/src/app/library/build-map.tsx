"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";

import { buildTopicMap } from "@/client/sdk.gen";
import type { DocumentOut, JobOut, TopicMapOut } from "@/client/types.gen";
import { JOB_LABEL, JobLine } from "@/components/job-status";
import { StepLabel, ThreadStep } from "@/components/thread";
import { Button } from "@/components/ui/button";
import {
  isActive,
  jobDoing,
  jobState,
  passageCounts,
  plural,
  problem,
  taggingTime,
} from "@/lib/library";
import { momentLabel } from "@/lib/time";

// The topic map: the local model tags the passages it hasn't read, then every tag in the
// library is grouped into topics. Questions are only written from tagged passages.

async function build(): Promise<TopicMapOut> {
  const { data } = await buildTopicMap({ throwOnError: true });
  return data;
}

/** The latest build: waiting, under way, or how it ended. */
function Build({ job, worker }: { job: JobOut; worker: boolean | undefined }) {
  const state = jobState(job, worker);
  const when = job.finished_at ? `${momentLabel(job.finished_at)} · ` : "";
  if (state === "done") {
    return (
      <JobLine state="done" label="Built">
        {when}
        {job.progress}
      </JobLine>
    );
  }
  if (state === "failed") {
    return (
      <div className="grid gap-1.5">
        <JobLine state="failed">
          {when}stopped at {job.progress ?? "the start"}
        </JobLine>
        {job.error && (
          <p className="max-w-[62ch] font-mono text-[11px] leading-normal wrap-anywhere text-fg-2">
            {job.error}
          </p>
        )}
        <p className="max-w-[62ch] text-small text-fg-2">
          The passages tagged before it failed stay tagged, so building again carries on from
          there.
        </p>
      </div>
    );
  }
  return <JobLine state={state}>{jobDoing(job, state)}</JobLine>;
}

export function BuildMap({
  documents,
  job,
  worker,
  topics,
}: {
  documents: DocumentOut[];
  job: JobOut | null;
  worker: boolean | undefined;
  topics: number | undefined;
}) {
  const queryClient = useQueryClient();
  const start = useMutation({
    mutationFn: build,
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: [{ _id: "listJobs" }] }),
  });
  const { passages, tagged } = passageCounts(documents);
  const untagged = passages - tagged;
  const building = isActive(job);
  const reading = documents.some((document) => isActive(document.latest_job));
  const answer = start.data;
  const meta = [
    topics !== undefined && plural(topics, "topic"),
    passages > 0 &&
      (untagged === 0 ? "every passage in it" : `${plural(untagged, "passage")} not in it yet`),
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <ThreadStep>
      <StepLabel as="h2" meta={meta || undefined}>
        Build the topic map
      </StepLabel>
      <p className="mt-3 max-w-[62ch] text-fg-2">
        The local model reads each passage it hasn&apos;t read yet, names the concepts in it and
        judges whether a question could be asked of it. Every tag in the library is then grouped
        into topics, so the same idea in a paper and in a notebook lands in one.
      </p>
      <div className="mt-5 flex flex-wrap items-center gap-x-5 gap-y-3">
        <Button
          onClick={() => start.mutate()}
          disabled={passages === 0 || building || start.isPending}
        >
          {passages > 0 && untagged === 0 ? "Build it again" : "Build the topic map"}
        </Button>
        <span className="font-mono text-[11px] leading-[1.4] tracking-[0.04em] text-fg-2">
          {passages === 0
            ? "add material first"
            : untagged > 0
              ? `${taggingTime(untagged)} for ${plural(untagged, "new passage")}`
              : "every passage is tagged: it only groups the tags again"}
        </span>
      </div>
      <div className="mt-4 grid gap-2 empty:hidden">
        {job && <Build job={job} worker={worker} />}
        {reading && (
          <p className="max-w-[62ch] text-small text-fg-2">
            A source is still to be read. The worker reads it first, so a build asked for now
            includes it.
          </p>
        )}
        {start.isError && (
          <p role="alert" className="text-small text-thread">
            {problem(start.error)}
          </p>
        )}
        {answer?.job === null && (
          <p className="text-small">There is nothing to build it from yet: add material first.</p>
        )}
      </div>
      {job && (
        <p role="status" className="sr-only">
          Topic map: {JOB_LABEL[jobState(job, worker)].toLowerCase()}
        </p>
      )}
    </ThreadStep>
  );
}

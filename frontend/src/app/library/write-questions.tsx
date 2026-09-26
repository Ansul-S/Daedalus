"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useState } from "react";

import { generateQuestions } from "@/client/sdk.gen";
import type { DocumentOut, GenerateIn, GenerateOut, JobOut } from "@/client/types.gen";
import { Field, Input, Select } from "@/components/field";
import { JOB_LABEL, JobLine } from "@/components/job-status";
import { StepLabel, ThreadStep } from "@/components/thread";
import { Button } from "@/components/ui/button";
import { fieldErrors } from "@/lib/api-errors";
import {
  DEFAULT_BATCH,
  isActive,
  jobDoing,
  jobState,
  MAX_BATCH,
  passageCounts,
  plural,
  problem,
} from "@/lib/library";
import { momentLabel } from "@/lib/time";

// A batch of questions, written from the passages no accepted question covers yet. The API
// plans the batch when it is asked, from the passages the topic map has tagged by then.

async function write(body: GenerateIn): Promise<GenerateOut> {
  const { data } = await generateQuestions({ body, throwOnError: true });
  return data;
}

/** The latest batch: waiting, under way, stopped for the day, or how it ended. */
function Batch({ job, worker }: { job: JobOut; worker: boolean | undefined }) {
  const state = jobState(job, worker);
  const when = job.finished_at ? `${momentLabel(job.finished_at)} · ` : "";
  if (state === "done") {
    return (
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
        <JobLine state="done" label="Written">
          {when}
          {job.progress}
        </JobLine>
        <Link href="/questions" className="thread-link text-small">
          See them in the question bank
        </Link>
      </div>
    );
  }
  if (state === "failed" || (state === "queued" && job.error)) {
    // A batch that ran out of the day's allowance goes back in the queue with the rest of its
    // questions still to write. A batch fails only when every question in it failed, and each
    // question's reason is kept with it rather than on the batch.
    return (
      <div className="grid gap-1.5">
        {state === "failed" ? (
          <JobLine state="failed">
            {when}
            {job.progress}
          </JobLine>
        ) : (
          <JobLine state="queued" label="Stopped for now">
            {job.progress} so far · the rest waits in the queue
          </JobLine>
        )}
        {job.error ? (
          <p className="max-w-[62ch] font-mono text-[11px] leading-normal wrap-anywhere text-fg-2">
            {job.error}
          </p>
        ) : (
          <p className="max-w-[62ch] text-small text-fg-2">
            Every question in it failed. The worker&apos;s output says why, question by question.
          </p>
        )}
      </div>
    );
  }
  return <JobLine state={state}>{jobDoing(job, state)}</JobLine>;
}

/** What the API made of the request: a batch planned, one already on its way, or nothing
 * left to ask about. */
function Answer({ answer, asked }: { answer: GenerateOut; asked: number }) {
  if (answer.job === null) {
    return (
      <p className="max-w-[62ch] text-small">
        Nothing is left to ask about: every passage worth a question already has an accepted
        one. Add material, or build the topic map first.
      </p>
    );
  }
  if (answer.message.startsWith("already")) {
    return (
      <p className="max-w-[62ch] text-small">
        A batch is already on its way, and one runs at a time: it carries on first.
      </p>
    );
  }
  if (answer.planned < asked) {
    return (
      <p className="max-w-[62ch] text-small">
        Planned {plural(answer.planned, "question")} of the {asked} asked for: that is every
        passage nothing has asked about yet.
      </p>
    );
  }
  return null;
}

export function WriteQuestions({
  documents,
  job,
  mapJob,
  worker,
  accepted,
}: {
  documents: DocumentOut[];
  job: JobOut | null;
  mapJob: JobOut | null;
  worker: boolean | undefined;
  accepted: number | undefined;
}) {
  const queryClient = useQueryClient();
  const [count, setCount] = useState(String(DEFAULT_BATCH));
  const [source, setSource] = useState("");
  const start = useMutation({
    mutationFn: write,
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: [{ _id: "listJobs" }] }),
  });
  const errors = fieldErrors(start.error);
  const wanted = Number(count);
  const valid = Number.isInteger(wanted) && wanted >= 1 && wanted <= MAX_BATCH;
  const writing = isActive(job);
  const mapping = isActive(mapJob);
  const { passages, tagged } = passageCounts(documents);
  const untagged = passages - tagged;
  const blocked = writing || mapping || tagged === 0;
  const sources = documents
    .filter((document) => (document.tagged_count ?? 0) > 0)
    .sort((a, b) => a.title.localeCompare(b.title));

  function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (blocked || start.isPending || !valid) return;
    start.mutate({ count: wanted, document_id: source ? Number(source) : null });
  }

  return (
    <ThreadStep>
      <StepLabel
        as="h2"
        meta={accepted === undefined ? undefined : `${plural(accepted, "question")} accepted so far`}
      >
        Write questions
      </StepLabel>
      <p className="mt-3 max-w-[62ch] text-fg-2">
        Each question is written from passages no question covers yet, by Groq&apos;s gpt-oss-120b
        on its free tier (Gemini, then the local model, when Groq is unavailable), and checked by
        the local model before it joins the library.
      </p>
      <form
        onSubmit={submit}
        aria-label="Write questions"
        className="mt-5 grid max-w-[46rem] grid-cols-1 items-end gap-x-4 gap-y-3 sm:grid-cols-[7rem_minmax(0,1fr)_auto]"
      >
        <Field label="How many" error={errors.get("count")}>
          {(control) => (
            <Input
              {...control}
              type="number"
              inputMode="numeric"
              min={1}
              max={MAX_BATCH}
              step={1}
              required
              value={count}
              onChange={(event) => setCount(event.target.value)}
            />
          )}
        </Field>
        <Field label="From" error={errors.get("document_id")}>
          {(control) => (
            <Select {...control} value={source} onChange={(event) => setSource(event.target.value)}>
              <option value="">Every source</option>
              {sources.map((document) => (
                <option key={document.id} value={document.id}>
                  {document.title}
                </option>
              ))}
            </Select>
          )}
        </Field>
        <Button
          type="submit"
          className="justify-self-start"
          disabled={blocked || start.isPending || !valid}
        >
          {valid ? `Write ${plural(wanted, "question")}` : "Write questions"}
        </Button>
      </form>
      <div className="mt-4 grid gap-2 empty:hidden">
        {tagged === 0 ? (
          <p className="max-w-[62ch] text-small text-fg-2">
            Questions are written from the topic map: add material and build it first.
          </p>
        ) : mapping ? (
          <p className="max-w-[62ch] text-small text-fg-2">
            Waiting for the topic map: a batch planned now would leave out the passages it is
            still tagging.
          </p>
        ) : (
          untagged > 0 && (
            <p className="max-w-[62ch] text-small text-fg-2">
              {plural(untagged, "passage")} {untagged === 1 ? "is" : "are"} not in the topic map
              yet, so a batch planned now leaves {untagged === 1 ? "it" : "them"} out. Build the
              map first to include {untagged === 1 ? "it" : "them"}.
            </p>
          )
        )}
        {job && <Batch job={job} worker={worker} />}
        {start.isError && errors.size === 0 && (
          <p role="alert" className="text-small text-thread">
            {problem(start.error)}
          </p>
        )}
        {start.data && <Answer answer={start.data} asked={start.variables?.count ?? wanted} />}
      </div>
      {job && (
        <p role="status" className="sr-only">
          Batch of questions: {JOB_LABEL[jobState(job, worker)].toLowerCase()}
        </p>
      )}
    </ThreadStep>
  );
}

/** The end of the thread: what the library is for. */
export function PractiseStep() {
  return (
    <ThreadStep end>
      <StepLabel as="h2" meta="on the practice page">
        Practise
      </StepLabel>
      <p className="mt-3 max-w-[62ch] text-fg-2">
        Accepted questions join practice straight away: what is due for review comes first, then
        something new from your weakest topic.
      </p>
      <Button asChild variant="outline" className="mt-5">
        <Link href="/practice">Go to practice</Link>
      </Button>
    </ThreadStep>
  );
}

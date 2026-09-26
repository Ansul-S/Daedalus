"use client";

import { type QueryClient, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";

import {
  listDocumentsOptions,
  listJobsOptions,
  listQuestionsOptions,
  listTopicsOptions,
  workerStatusOptions,
} from "@/client/@tanstack/react-query.gen";
import type { DocumentOut, JobOut } from "@/client/types.gen";
import { ApiProblem } from "@/components/api-problem";
import { Commands } from "@/components/commands";
import { JobMark } from "@/components/job-status";
import { SheetSection, TitleBlock } from "@/components/sheet";
import { Thread } from "@/components/thread";
import { isActive, passageCounts, plural } from "@/lib/library";
import { number } from "@/lib/progress";

import { AddMaterial } from "./add-material";
import { BuildMap } from "./build-map";
import { Shelves } from "./shelves";
import { PractiseStep, WriteQuestions } from "./write-questions";

// The library as a thread of work: add material, build the topic map over it, write questions
// from it, then practise them. While a job waits or runs, the page follows it every two
// seconds, and fetches again whatever the job changes as it goes.

const FOLLOW_MS = 2000;
// How often to ask whether a worker has started or stopped
const WORKER_MS = 5000;

// What each kind of job changes while it runs, and once more when it ends
const CHANGES: Record<string, { going: string[]; ended: string[] }> = {
  topics: { going: ["listDocuments"], ended: ["listDocuments", "listTopics"] },
  generate: {
    going: ["listQuestions", "listTopics"],
    ended: ["listQuestions", "listTopics", "practiceNext", "practiceMap", "practiceProgress"],
  },
};

function refresh(queryClient: QueryClient, kind: string, ended: boolean) {
  const changes = CHANGES[kind];
  if (!changes) return;
  for (const _id of ended ? changes.ended : changes.going) {
    void queryClient.invalidateQueries({ queryKey: [{ _id }] });
  }
}

/** Follows a job: each change in its state or progress line fetches again what it changes. */
function useFollow(job: JobOut | null) {
  const queryClient = useQueryClient();
  const seen = useRef<string | null>(null);
  const key = job ? `${job.id}:${job.status}:${job.progress}` : null;
  const kind = job?.kind ?? null;
  const ended = job !== null && !isActive(job);

  useEffect(() => {
    const before = seen.current;
    seen.current = key;
    if (before === null || key === null || kind === null || before === key) return;
    refresh(queryClient, kind, ended);
  }, [key, kind, ended, queryClient]);
}

/** Polls while the query's own data shows work waiting or under way. */
function whileActive<T>(active: (data: T) => boolean) {
  return ({ state }: { state: { data: T | undefined } }) =>
    state.data !== undefined && active(state.data) ? FOLLOW_MS : false;
}

const newest = (jobs: JobOut[]) => isActive(jobs[0]);

function WorkerNotice({ running, waiting }: { running: boolean | undefined; waiting: number }) {
  if (running !== false) return null;
  if (waiting === 0) {
    return (
      <p className="type-label mb-8 flex flex-wrap items-center gap-x-2 gap-y-1 text-fg-2">
        <JobMark state="queued" />
        No worker running · what you ask for here waits for{" "}
        <code className="font-mono normal-case">make worker</code>
      </p>
    );
  }
  return (
    <div role="status" className="mb-10 grid grid-cols-1 gap-4 border-l-2 border-thread pl-4">
      <p className="max-w-[62ch]">
        <strong className="font-semibold">No worker is running,</strong> so{" "}
        {waiting === 1 ? "a job waits" : `${plural(waiting, "job")} wait`} in the queue. Start one
        from the repository and it takes them in turn, picking up any job that was stopped part
        way.
      </p>
      <Commands
        title="Starting the worker"
        lines={[{ command: "make worker", comment: "Ctrl+C stops it" }]}
      />
    </div>
  );
}

export function Library({ children }: { children: React.ReactNode }) {
  const documents = useQuery({
    ...listDocumentsOptions(),
    refetchInterval: whileActive((found: DocumentOut[]) =>
      found.some(({ latest_job }) => isActive(latest_job)),
    ),
  });
  const maps = useQuery({
    ...listJobsOptions({ query: { kind: "topics", limit: 1 } }),
    refetchInterval: whileActive(newest),
  });
  const batches = useQuery({
    ...listJobsOptions({ query: { kind: "generate", limit: 1 } }),
    refetchInterval: whileActive(newest),
  });
  const worker = useQuery({ ...workerStatusOptions(), refetchInterval: WORKER_MS });
  const topics = useQuery(listTopicsOptions({ query: { limit: 500 } }));
  const accepted = useQuery(listQuestionsOptions({ query: { status: "accepted", limit: 1 } }));

  const mapJob = maps.data?.[0] ?? null;
  const batchJob = batches.data?.[0] ?? null;
  useFollow(mapJob);
  useFollow(batchJob);

  const found = documents.data;
  const running = worker.data?.running;
  const counts = found ? passageCounts(found) : null;
  const waiting = [...(found ?? []).map((document) => document.latest_job), mapJob, batchJob].filter(
    isActive,
  ).length;

  return (
    <>
      <SheetSection aria-busy={documents.isFetching && !found}>
        {children}
        {found ? (
          <>
            <WorkerNotice running={running} waiting={waiting} />
            <Thread aria-label="From material to practice" className="pt-2">
              <AddMaterial documents={found} worker={running} />
              <BuildMap
                documents={found}
                job={mapJob}
                worker={running}
                topics={topics.data?.length}
              />
              <WriteQuestions
                documents={found}
                job={batchJob}
                mapJob={mapJob}
                worker={running}
                accepted={accepted.data?.total}
              />
              <PractiseStep />
            </Thread>
          </>
        ) : documents.isError ? (
          <ApiProblem error={documents.error} retry={() => void documents.refetch()} />
        ) : (
          <p className="text-fg-2">Opening the library…</p>
        )}
      </SheetSection>
      {found && (
        <SheetSection aria-labelledby="shelves">
          <Shelves documents={found} worker={running} />
        </SheetSection>
      )}
      <TitleBlock
        cells={[
          { label: "Sheet", value: "L-01" },
          { label: "Drawing", value: "Library" },
          { label: "Documents", value: found ? number(found.length) : "–" },
          { label: "Passages", value: counts ? number(counts.passages) : "–" },
          { label: "In the map", value: counts ? number(counts.tagged) : "–" },
          { label: "Topics", value: topics.data ? number(topics.data.length) : "–" },
          {
            label: "Worker",
            value: running === undefined ? "–" : running ? "Running" : "Stopped",
          },
        ]}
      />
    </>
  );
}

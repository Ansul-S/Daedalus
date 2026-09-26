import type { DocumentOut, JobOut } from "@/client/types.gen";
import { ApiError } from "@/lib/api-errors";
import { number } from "@/lib/progress";

// The library's work: a document read into passages, the topic map built over them, a batch
// of questions written from them. Each is a job in the API's queue until a worker takes it.

/** How many questions a batch may ask for (backend/app/api/questions.py). */
export const MAX_BATCH = 100;
export const DEFAULT_BATCH = 10;

/** The files the API reads (backend/app/ingest/storage.py). */
export const FILE_TYPES = [".pdf", ".ipynb"];

// The local model tags a passage in about 12 seconds (README, "The topic map")
const TAGGING_SECONDS = 12;

export type JobState = "queued" | "running" | "interrupted" | "done" | "failed";

export function isActive(job: JobOut | null | undefined): job is JobOut {
  return job?.status === "queued" || job?.status === "running";
}

/** Where a job stands. A job still marked running while no worker is running was stopped part
 * way, and the next worker to start queues it again. `worker` is unknown until the API says. */
export function jobState(job: JobOut, worker: boolean | undefined): JobState {
  switch (job.status) {
    case "running":
      return worker === false ? "interrupted" : "running";
    case "done":
    case "failed":
      return job.status;
    default:
      return "queued";
  }
}

/** What a job waiting in the queue, or under way, is doing. */
export function jobDoing(job: JobOut, state: JobState): string {
  if (state === "running") return job.progress ?? "starting";
  if (state === "interrupted") {
    return `${job.progress ?? "started"} · picked up again when a worker starts`;
  }
  return "waiting for the worker";
}

/** 1 passage, 1,284 passages */
export function plural(count: number, noun: string): string {
  return `${number(count)} ${noun}${count === 1 ? "" : "s"}`;
}

/** The passages of the whole library, and how many of them the topic map has tagged. */
export function passageCounts(documents: DocumentOut[]): { passages: number; tagged: number } {
  return documents.reduce(
    (sum, document) => ({
      passages: sum.passages + (document.chunk_count ?? 0),
      tagged: sum.tagged + (document.tagged_count ?? 0),
    }),
    { passages: 0, tagged: 0 },
  );
}

/** About 3 minutes: how long the local model takes over this many passages. */
export function taggingTime(passages: number): string {
  const minutes = Math.ceil((passages * TAGGING_SECONDS) / 60);
  return minutes <= 1 ? "about a minute" : `about ${minutes} minutes`;
}

/** Whether a file's name says it is one the API reads. */
export function readable(file: File): boolean {
  const name = file.name.toLowerCase();
  return FILE_TYPES.some((type) => name.endsWith(type));
}

/** 12 KB, 2.1 MB */
export function fileSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** arXiv · 1706.03762 v7 · cs.CL: the kind of source, then which paper or file it is. */
export function sourceParts(document: DocumentOut): string[] {
  if (document.source_type === "arxiv") {
    const categories = document.details.categories;
    return [
      "arXiv",
      [document.arxiv_id, document.arxiv_version].filter(Boolean).join(" "),
      ...(Array.isArray(categories) ? categories.filter((c) => typeof c === "string") : []),
    ];
  }
  const kind = document.source_type === "pdf" ? "PDF" : "Notebook";
  return document.filename ? [kind, document.filename] : [kind];
}

/** 11 pages, 223 cells: how long the source is, when the parser counted it. */
export function sourceLength(document: DocumentOut): string | null {
  const { pages, cells } = document.details;
  if (typeof pages === "number") return plural(pages, "page");
  if (typeof cells === "number") return plural(cells, "cell");
  return null;
}

/** Up to four authors; past that, the first three and how many more there are. */
export function authorsLine(authors: string): string {
  const names = authors.split(", ");
  if (names.length <= 4) return authors;
  return `${names.slice(0, 3).join(", ")} and ${names.length - 3} more`;
}

/** Why the API refused, in a few words. */
export function problem(error: unknown): string {
  if (error instanceof ApiError && error.status === null) return "The API can't be reached.";
  const message = error instanceof Error ? error.message : "The API refused the request";
  const sentence = message.charAt(0).toUpperCase() + message.slice(1);
  return /[.!?]$/.test(sentence) ? sentence : `${sentence}.`;
}

"use client";

import { useQueries } from "@tanstack/react-query";
import Link from "next/link";

import { listQuestionsOptions } from "@/client/@tanstack/react-query.gen";
import type { DocumentOut } from "@/client/types.gen";
import { Commands } from "@/components/commands";
import { Disclosure } from "@/components/disclosure";
import {
  authorsLine,
  isActive,
  passageCounts,
  plural,
  sourceLength,
  sourceParts,
} from "@/lib/library";
import { number } from "@/lib/progress";
import { dayOfMoment } from "@/lib/time";
import { cn } from "@/lib/utils";

import { DocumentState } from "./document-state";

// Every document in the library, newest first: what it is, how its reading went, how much of
// it the topic map has read, and the questions written from it.

const MONO = "font-mono text-[11px] leading-[1.4] tracking-[0.04em]";
// A separator that stays at the end of the line when the line wraps
const SEP = "\u00a0· ";

/** How much of a document the topic map has tagged. */
function inTheMap(document: DocumentOut): string | null {
  const { chunk_count: passages = 0, tagged_count: tagged = 0 } = document;
  if (passages === 0) return null;
  if (tagged === passages) return "all in the topic map";
  if (tagged === 0) return "not in the topic map yet";
  return `${number(tagged)} in the topic map`;
}

function Questions({ id, count }: { id: number; count: number | undefined }) {
  if (count === undefined) return <>questions –</>;
  if (count === 0) return <>no questions yet</>;
  return (
    <Link href={`/questions?source=${id}`} className="thread-link">
      {plural(count, "question")}
    </Link>
  );
}

function DocumentRow({
  document,
  questions,
  worker,
}: {
  document: DocumentOut;
  questions: number | undefined;
  worker: boolean | undefined;
}) {
  const reading = document.latest_job;
  // A document read before keeps its passages when a later reading fails; while it is read
  // again, the old failure no longer stands
  const failure =
    reading?.status === "failed"
      ? reading.error
      : document.status === "failed" && !isActive(reading)
        ? document.error
        : null;
  return (
    <li className="grid grid-cols-1 gap-x-6 gap-y-1.5 border-b border-dotted border-line-2 py-4 wrap-anywhere sm:grid-cols-[minmax(0,1fr)_auto]">
      <p className={cn(MONO, "text-fg-2 uppercase")}>{sourceParts(document).join(SEP)}</p>
      <div className="sm:col-start-2 sm:row-start-1 sm:justify-self-end">
        <DocumentState
          document={document}
          worker={worker}
          ready={`added ${dayOfMoment(document.created_at)}`}
        />
      </div>
      <h3 className="max-w-[48rem] text-[1.0625rem] leading-[1.45] sm:col-span-2">
        {document.url ? (
          <a href={document.url} target="_blank" rel="noreferrer" className="thread-link">
            {document.title}
            <span aria-hidden> ↗</span>
            <span className="sr-only"> (opens in a new tab)</span>
          </a>
        ) : (
          document.title
        )}
      </h3>
      {document.authors && (
        <p className="text-small text-fg-2 sm:col-span-2">{authorsLine(document.authors)}</p>
      )}
      <p className={cn(MONO, "text-fg-2 sm:col-span-2")}>
        {[sourceLength(document), plural(document.chunk_count ?? 0, "passage"), inTheMap(document)]
          .filter(Boolean)
          .join(SEP)}
        {SEP}
        <Questions id={document.id} count={questions} />
      </p>
      {failure && (
        <Disclosure
          summary={document.status === "ready" ? "The latest reading failed" : "Why it failed"}
          className="mt-1 sm:col-span-2"
        >
          <p className="font-mono text-[11px] leading-normal">{failure}</p>
        </Disclosure>
      )}
    </li>
  );
}

export function Shelves({
  documents,
  worker,
}: {
  documents: DocumentOut[];
  worker: boolean | undefined;
}) {
  // The accepted questions written from each document
  const counts = useQueries({
    queries: documents.map((document) =>
      listQuestionsOptions({
        query: { document_id: document.id, status: "accepted", limit: 1 },
      }),
    ),
  });
  const { passages } = passageCounts(documents);

  return (
    <>
      <header className="mb-5 flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <h2 id="shelves" className="type-h3">
          The shelves
        </h2>
        <p className="type-label text-fg-2">
          {documents.length === 0
            ? "empty"
            : `${plural(documents.length, "document")} · ${plural(passages, "passage")}`}
        </p>
      </header>
      {documents.length === 0 ? (
        <>
          <p className="mb-5 max-w-[62ch]">
            Nothing here yet. Add a PDF, a notebook or an arXiv paper above, or whole folders from
            the command line:
          </p>
          <Commands
            title="Adding material"
            lines={[{ command: 'make ingest SRC="notes/ 1706.03762"' }]}
          />
        </>
      ) : (
        <ul className="border-t border-line-2">
          {documents.map((document, i) => (
            <DocumentRow
              key={document.id}
              document={document}
              questions={counts[i]?.data?.total}
              worker={worker}
            />
          ))}
        </ul>
      )}
    </>
  );
}

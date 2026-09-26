"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useId, useRef, useState } from "react";

import { addArxivPaper, uploadDocument } from "@/client/sdk.gen";
import type { DocumentOut, IngestOut } from "@/client/types.gen";
import { Checkbox, Field, Input } from "@/components/field";
import { FileDrop } from "@/components/file-drop";
import { JobLine } from "@/components/job-status";
import { StepLabel, ThreadStep } from "@/components/thread";
import { Button } from "@/components/ui/button";
import { ApiError } from "@/lib/api-errors";
import { FILE_TYPES, fileSize, passageCounts, plural, problem, readable } from "@/lib/library";
import { cn } from "@/lib/utils";

import { DocumentState } from "./document-state";

// Adding material: files uploaded one after another, or an arXiv paper by its ID. Each becomes
// a document in the queue for the worker to read, and shows here how that reading goes.

type Reading = { ocr: boolean; formulas: boolean; force: boolean };

const READING: Reading = { ocr: false, formulas: true, force: false };
// A file name, an address or an error can be one long word: the card keeps its width, and the
// word breaks, or is cut short in the list to upload
const CARD =
  "grid min-w-0 grid-cols-1 content-start gap-4 border border-line-2 px-[18px] pt-4 pb-[18px] wrap-anywhere";
const MONO = "font-mono text-[11px] leading-[1.4] tracking-[0.04em]";

/** One file or paper sent to the API: the document it became, or why it was refused. */
type Added = { key: number; name: string; out: IngestOut | null; error: string | null };

function sameFile(a: File, b: File): boolean {
  return a.name === b.name && a.size === b.size && a.lastModified === b.lastModified;
}

function ReadingOptions({
  value,
  onChange,
  pdf,
}: {
  value: Reading;
  onChange: (next: Reading) => void;
  pdf: string;
}) {
  return (
    <fieldset className="grid gap-1.5">
      <legend className="type-label mb-2 text-fg-2">Reading</legend>
      <Checkbox
        label="Formulas as LaTeX"
        hint={`${pdf}; slower`}
        checked={value.formulas}
        onChange={(event) => onChange({ ...value, formulas: event.target.checked })}
      />
      <Checkbox
        label="Scanned pages (OCR)"
        hint={pdf}
        checked={value.ocr}
        onChange={(event) => onChange({ ...value, ocr: event.target.checked })}
      />
      <Checkbox
        label="Read it again"
        hint="even if it is in the library"
        checked={value.force}
        onChange={(event) => onChange({ ...value, force: event.target.checked })}
      />
    </fieldset>
  );
}

/** How the reading asked for is getting on. One that failed says why, even when the passages
 * of an earlier reading keep the document ready. */
function SentState({ document, worker }: { document: DocumentOut; worker: boolean | undefined }) {
  const job = document.latest_job;
  if (job?.status !== "failed") return <DocumentState document={document} worker={worker} />;
  return (
    <>
      <JobLine state="failed">
        {document.status === "ready" ? "the passages read before stay in the library" : null}
      </JobLine>
      {job.error && <p className={cn(MONO, "text-fg-2")}>{job.error}</p>}
    </>
  );
}

/** What was sent from one card, each with how its reading is getting on. */
function AddedList({
  added,
  documents,
  worker,
}: {
  added: Added[];
  documents: DocumentOut[];
  worker: boolean | undefined;
}) {
  if (added.length === 0) return null;
  return (
    <ul aria-label="Sent" className="border-t border-line-2">
      {added.map(({ key, name, out, error }) => {
        // The document as last fetched: its title, once read, and its latest job. Until the
        // list is fetched again, the one the API sent back has the newer job.
        const listed = out && documents.find(({ id }) => id === out.document.id);
        const document =
          out &&
          (listed && (listed.latest_job?.id ?? 0) >= (out.job?.id ?? 0) ? listed : out.document);
        return (
          <li key={key} className="grid gap-1 border-b border-dotted border-line-2 py-2.5">
            <span className="text-small">{document?.title ?? name}</span>
            {document && document.title !== name && (
              <span className={cn(MONO, "text-fg-2")}>{name}</span>
            )}
            {error ? (
              <p className="text-small text-thread">{error}</p>
            ) : out?.job === null ? (
              <JobLine state="done" label="In the library">
                read before · tick Read it again to read it anew
              </JobLine>
            ) : (
              document && <SentState document={document} worker={worker} />
            )}
          </li>
        );
      })}
    </ul>
  );
}

function Files({ documents, worker }: { documents: DocumentOut[]; worker: boolean | undefined }) {
  const queryClient = useQueryClient();
  const [chosen, setChosen] = useState<File[]>([]);
  const [leftOut, setLeftOut] = useState<string[]>([]);
  const [reading, setReading] = useState(READING);
  const [added, setAdded] = useState<Added[]>([]);
  const [sending, setSending] = useState(false);
  const count = useRef(0);
  const title = useId();

  function choose(files: File[]) {
    setLeftOut(files.filter((file) => !readable(file)).map((file) => file.name));
    setChosen((current) => [
      ...current,
      ...files.filter((file) => readable(file) && !current.some((kept) => sameFile(kept, file))),
    ]);
  }

  // One file at a time, so each result shows as it comes
  async function upload(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (sending || chosen.length === 0) return;
    setSending(true);
    setLeftOut([]);
    for (const file of chosen) {
      const key = count.current++;
      let result: Added;
      try {
        const { data } = await uploadDocument({
          body: { file },
          query: reading,
          throwOnError: true,
        });
        result = { key, name: file.name, out: data, error: null };
      } catch (error) {
        result = { key, name: file.name, out: null, error: problem(error) };
      }
      setAdded((current) => [...current, result]);
      setChosen((current) => current.filter((kept) => kept !== file));
      void queryClient.invalidateQueries({ queryKey: [{ _id: "listDocuments" }] });
    }
    setSending(false);
  }

  return (
    <form onSubmit={upload} aria-labelledby={title} className={CARD}>
      <h3 id={title} className="type-label text-fg-2">
        Files · PDF or Jupyter notebook
      </h3>
      <FileDrop accept={FILE_TYPES} onFiles={choose} disabled={sending}>
        Drop PDFs or notebooks here, or
      </FileDrop>
      {leftOut.length > 0 && (
        <p role="alert" className="text-small text-thread">
          Left out, not a PDF or a notebook: {leftOut.join(", ")}
        </p>
      )}
      {chosen.length > 0 && (
        <ul aria-label="To upload" className="border-t border-line-2">
          {chosen.map((file) => (
            <li
              key={`${file.name}:${file.size}:${file.lastModified}`}
              className={cn(MONO, "flex items-center gap-3 border-b border-dotted border-line-2 py-1")}
            >
              <span className="min-w-0 flex-1 truncate">{file.name}</span>
              <span className="text-fg-2">{fileSize(file.size)}</span>
              <button
                type="button"
                aria-label={`Leave out ${file.name}`}
                disabled={sending}
                onClick={() => setChosen((current) => current.filter((kept) => kept !== file))}
                className="cursor-pointer px-1.5 py-1 text-sm leading-none text-fg-2 hover:text-fg disabled:opacity-50"
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
      <ReadingOptions value={reading} onChange={setReading} pdf="PDFs only" />
      <Button
        type="submit"
        size="sm"
        className="justify-self-start"
        disabled={sending || chosen.length === 0}
      >
        {sending
          ? "Uploading…"
          : chosen.length > 0
            ? `Upload ${plural(chosen.length, "file")}`
            : "Upload"}
      </Button>
      <AddedList added={added} documents={documents} worker={worker} />
    </form>
  );
}

function Paper({ documents, worker }: { documents: DocumentOut[]; worker: boolean | undefined }) {
  const queryClient = useQueryClient();
  const [wanted, setWanted] = useState("");
  const [reading, setReading] = useState(READING);
  const [added, setAdded] = useState<Added[]>([]);
  const [refused, setRefused] = useState<{ field: boolean; message: string } | null>(null);
  const [sending, setSending] = useState(false);
  const field = useRef<HTMLInputElement>(null);
  const count = useRef(0);
  const title = useId();

  async function add(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const arxivId = wanted.trim();
    if (sending || !arxivId) return;
    setSending(true);
    setRefused(null);
    try {
      const { data } = await addArxivPaper({
        body: { arxiv_id: arxivId, ...reading },
        throwOnError: true,
      });
      setAdded((current) => [...current, { key: count.current++, name: arxivId, out: data, error: null }]);
      setWanted("");
      void queryClient.invalidateQueries({ queryKey: [{ _id: "listDocuments" }] });
    } catch (error) {
      // A 422 is about what was typed; anything else is about the API
      const typed = error instanceof ApiError && error.status === 422;
      setRefused({ field: typed, message: problem(error) });
      if (typed) field.current?.focus();
    }
    setSending(false);
  }

  return (
    <form onSubmit={add} aria-labelledby={title} className={CARD}>
      <h3 id={title} className="type-label text-fg-2">
        arXiv paper · read from its HTML
      </h3>
      <Field
        label="ID or address"
        hint="1706.03762, or 1706.03762v7 for one version; its arxiv.org address works too"
        error={refused?.field ? refused.message : null}
      >
        {(control) => (
          <Input
            {...control}
            ref={field}
            value={wanted}
            onChange={(event) => setWanted(event.target.value)}
            placeholder="1706.03762"
            autoComplete="off"
            spellCheck={false}
          />
        )}
      </Field>
      <ReadingOptions value={reading} onChange={setReading} pdf="only for a paper without HTML" />
      <Button
        type="submit"
        size="sm"
        className="justify-self-start"
        disabled={sending || !wanted.trim()}
      >
        {sending ? "Adding…" : "Add paper"}
      </Button>
      {refused && !refused.field && (
        <p role="alert" className="text-small text-thread">
          {refused.message}
        </p>
      )}
      <AddedList added={added} documents={documents} worker={worker} />
    </form>
  );
}

export function AddMaterial({
  documents,
  worker,
}: {
  documents: DocumentOut[];
  worker: boolean | undefined;
}) {
  const { passages } = passageCounts(documents);
  return (
    <ThreadStep>
      <StepLabel
        as="h2"
        meta={
          documents.length === 0
            ? "a PDF, a notebook or an arXiv paper"
            : `${plural(documents.length, "document")} · ${plural(passages, "passage")}`
        }
      >
        Add material
      </StepLabel>
      <p className="mt-3 max-w-[62ch] text-fg-2">
        The worker reads each source into passages, and keeps the section, page or cell of each so
        that a question can cite it.
      </p>
      <div className="mt-5 grid grid-cols-1 items-start gap-5 lg:grid-cols-2">
        <Files documents={documents} worker={worker} />
        <Paper documents={documents} worker={worker} />
      </div>
    </ThreadStep>
  );
}

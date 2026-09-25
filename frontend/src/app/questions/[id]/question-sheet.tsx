"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useEffect, useId, useRef, useState } from "react";
import { toast } from "sonner";

import { getQuestionOptions } from "@/client/@tanstack/react-query.gen";
import type { QuestionDetailOut, SourceOut } from "@/client/types.gen";
import { ApiProblem } from "@/components/api-problem";
import { Disclosure } from "@/components/disclosure";
import { Pips } from "@/components/hatching";
import { Markdown } from "@/components/markdown";
import { Rate } from "@/components/rating";
import { SheetHead, SheetSection, TitleBlock } from "@/components/sheet";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ApiError } from "@/lib/api-errors";
import { questionNumber, statusLabel, styleLabel } from "@/lib/questions";
import { readReport, type Report } from "@/lib/report";
import { dateLabel } from "@/lib/time";
import { cn } from "@/lib/utils";

import { EditForm, type EditOutcome } from "./edit-form";
import { Checks, Edits, Origin } from "./report";
import { type StatusAction, StatusChange } from "./status-change";

// One question and everything behind it: the passages it was written from, what an answer has
// to cover with the quote that proves each point, the checks it went through and every
// correction since. It can be corrected, retired or put back, and rated.

const MONO = "font-mono text-[11px] leading-[1.4] tracking-[0.04em]";

function Crumbs({ id }: { id: number }) {
  return (
    <nav aria-label="Breadcrumb" className="type-label mb-7 text-fg-2">
      <Link href="/questions" className="thread-link">
        Questions
      </Link>
      <span aria-hidden> / </span>
      <span aria-current="page">{questionNumber(id)}</span>
    </nav>
  );
}

function Part({
  title,
  className,
  children,
}: {
  title: React.ReactNode;
  className?: string;
  children: React.ReactNode;
}) {
  const id = useId();
  return (
    <section aria-labelledby={id} className={cn("border-t border-line-2 pt-4", className)}>
      <h2 id={id} className="type-label mb-3.5 text-fg-2">
        {title}
      </h2>
      {children}
    </section>
  );
}

function plural(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

function Standing({ question, report }: { question: QuestionDetailOut; report: Report }) {
  return (
    <p className="flex flex-wrap gap-1.5">
      {question.status === "accepted" && <Badge>Accepted · in practice</Badge>}
      {question.status === "retired" && <Badge variant="dashed">Retired · out of practice</Badge>}
      {question.status === "rejected" && (
        <Badge variant="dashed">
          Rejected{report.failed.length > 0 && ` · ${report.failed.join(", ")}`}
        </Badge>
      )}
      {question.source_updated && <Badge variant="thread">Source updated</Badge>}
    </p>
  );
}

function passageName(sources: SourceOut[], chunkId: number | null | undefined): string {
  const index = sources.findIndex((source) => source.chunk_id === chunkId);
  return index === -1 ? "not one of its passages" : `passage ${index + 1}`;
}

function KeyPoints({ question }: { question: QuestionDetailOut }) {
  return (
    <ol className="max-w-[60rem]">
      {question.key_points.map((point, i) => (
        <li
          key={i}
          className="grid grid-cols-[2.25rem_minmax(0,1fr)] gap-x-3 border-b border-dotted border-line-2 py-3 first:pt-0 last:border-b-0"
        >
          <span className={cn(MONO, "pt-1 text-fg-2 uppercase")}>k{i + 1}</span>
          <div className="grid min-w-0 gap-1.5">
            <Markdown inline>{point.text}</Markdown>
            <p className={cn(MONO, "text-fg-2 uppercase")}>
              weight <Pips value={point.weight} of={3} label="weight" /> ·{" "}
              {passageName(question.sources, point.chunk_id)}
            </p>
            <blockquote className="border-l-2 border-line-2 pl-3 text-small text-fg-2">
              <span className="sr-only">The passage says: </span>“
              <Markdown inline className="italic">
                {point.evidence_quote}
              </Markdown>
              ”
            </blockquote>
          </div>
        </li>
      ))}
    </ol>
  );
}

function Sources({ sources }: { sources: SourceOut[] }) {
  return (
    <ol className="grid max-w-[60rem] gap-5">
      {sources.map((source, i) => (
        <li key={source.chunk_id} className="grid grid-cols-[2.25rem_minmax(0,1fr)] gap-x-3">
          <span className={cn(MONO, "pt-1 text-fg-2")}>[{i + 1}]</span>
          <div className="grid min-w-0 gap-2.5">
            <p className="flex flex-wrap items-center gap-x-3 gap-y-1.5 text-small">
              {source.link ? (
                <a href={source.link} target="_blank" rel="noreferrer" className="thread-link">
                  {source.citation}
                </a>
              ) : (
                source.citation
              )}
              {source.superseded && <Badge variant="thread">Replaced by a later ingestion</Badge>}
            </p>
            <Disclosure summary="Read the passage">
              <Markdown className="text-small">{source.text}</Markdown>
            </Disclosure>
          </div>
        </li>
      ))}
    </ol>
  );
}

function Loaded({ question }: { question: QuestionDetailOut }) {
  const [editing, setEditing] = useState(false);
  const [action, setAction] = useState<StatusAction | null>(null);
  const editButton = useRef<HTMLButtonElement>(null);
  const statusButton = useRef<HTMLButtonElement>(null);
  const refocus = useRef<"edit" | "status" | null>(null);
  const report = readReport(question.validation);
  const name = questionNumber(question.id);
  const rejected = question.status === "rejected";

  // Back to the button that opened the form, once the form has gone
  useEffect(() => {
    if (editing || action !== null || refocus.current === null) return;
    (refocus.current === "edit" ? editButton : statusButton).current?.focus();
    refocus.current = null;
  }, [editing, action]);

  function doneEditing(outcome: EditOutcome) {
    refocus.current = "edit";
    setEditing(false);
    if (outcome === "saved") {
      toast(`${name} corrected`, { description: "The change is kept with what it replaced." });
    } else if (outcome === "unchanged") {
      toast("Nothing to save", { description: "The form says what the question already says." });
    }
  }

  function doneChanging(changed: boolean) {
    refocus.current = "status";
    if (changed) {
      toast(
        action === "retire" ? `${name} retired` : `${name} is back in practice`,
        action === "retire" ? { description: "Put it back at any time." } : undefined,
      );
    }
    setAction(null);
  }

  return (
    <>
      <SheetSection>
        <Crumbs id={question.id} />
        <SheetHead number="Sheet Q-02" title={name} sigil="ζ" />
        <div className="grid gap-10">
          <div className="grid gap-4">
            <p className={cn(MONO, "text-fg-2 uppercase")}>
              {question.topic ?? "no topic"} · {styleLabel(question.style)} · difficulty{" "}
              <Pips value={question.difficulty} label="difficulty" /> · written{" "}
              {dateLabel(question.created_at.slice(0, 10))}
            </p>
            <Standing question={question} report={report} />
            {!editing && (
              <Markdown className="type-q max-w-[34em]">{question.text}</Markdown>
            )}
            {question.source_updated && (
              <p className={cn(MONO, "text-thread")}>
                A passage was ingested again after this question was written: check it still
                holds.
              </p>
            )}
            {!editing && (
              <div className="mt-2 flex max-w-[60rem] flex-wrap items-start justify-between gap-x-10 gap-y-4">
                <Rate rated="question" id={question.id} initial={question.rating} />
                {!rejected && action === null && (
                  <div className="flex flex-wrap gap-2">
                    <Button
                      ref={editButton}
                      variant="outline"
                      size="sm"
                      onClick={() => setEditing(true)}
                    >
                      Edit
                    </Button>
                    {question.status === "accepted" ? (
                      <Button
                        ref={statusButton}
                        variant="destructive"
                        size="sm"
                        onClick={() => setAction("retire")}
                      >
                        Retire
                      </Button>
                    ) : (
                      <Button
                        ref={statusButton}
                        variant="outline"
                        size="sm"
                        onClick={() => setAction("restore")}
                      >
                        Put back
                      </Button>
                    )}
                  </div>
                )}
              </div>
            )}
            {action !== null && (
              <StatusChange question={question} action={action} onClose={doneChanging} />
            )}
            {rejected && (
              <p className="max-w-[62ch] text-small text-fg-2">
                The checks turned this question down. It is kept as the record of why, so it
                can&apos;t be corrected or put in practice; rating it says whether the checks
                were right.
              </p>
            )}
          </div>

          {editing ? (
            <EditForm question={question} onDone={doneEditing} />
          ) : (
            <>
              <Part title="Reference answer">
                <Markdown className="max-w-[48rem]">{question.reference_answer}</Markdown>
              </Part>
              <Part title={`Key points · ${question.key_points.length}`}>
                <KeyPoints question={question} />
              </Part>
            </>
          )}
          {question.misconceptions.length > 0 && (
            <Part title="Misconceptions worth recognizing">
              <ul className="grid max-w-[60rem] list-[square] gap-1.5 pl-5 text-small marker:text-fg-2">
                {question.misconceptions.map((item, i) => (
                  <li key={i}>
                    <Markdown inline>{item}</Markdown>
                  </li>
                ))}
              </ul>
            </Part>
          )}
          <Part title={`Sources · ${plural(question.sources.length, "passage")}`}>
            <Sources sources={question.sources} />
          </Part>
          <Part title="Checks · when it was written">
            <Checks report={report} />
          </Part>
          <Part title={`Edits · ${report.edits.length}`}>
            <Edits edits={report.edits} />
          </Part>
          <Part title="Written by">
            <Origin question={question} />
          </Part>
        </div>
      </SheetSection>
      <TitleBlock
        cells={[
          { label: "Sheet", value: "Q-02" },
          { label: "Drawing", value: name },
          { label: "Status", value: statusLabel(question.status) },
          { label: "Topic", value: question.topic ?? "–" },
          {
            label: "Your rating",
            value: question.rating ? (question.rating.value === 1 ? "Good" : "Poor") : "–",
          },
          { label: "Edits", value: report.edits.length },
        ]}
      />
    </>
  );
}

export function QuestionSheet({ id }: { id: number }) {
  const found = useQuery(getQuestionOptions({ path: { question_id: id } }));
  const question = found.data;
  if (question) return <Loaded key={question.id} question={question} />;

  const missing = found.error instanceof ApiError && found.error.status === 404;
  return (
    <>
      <SheetSection aria-busy={found.isFetching}>
        <Crumbs id={id} />
        <SheetHead number="Sheet Q-02" title={questionNumber(id)} sigil="ζ" />
        {missing ? (
          <p className="max-w-[62ch]">
            There is no question {questionNumber(id)}.{" "}
            <Link href="/questions" className="thread-link">
              Back to the questions
            </Link>
          </p>
        ) : found.isError ? (
          <ApiProblem error={found.error} retry={() => void found.refetch()} />
        ) : (
          <p className="text-fg-2">Opening {questionNumber(id)}…</p>
        )}
      </SheetSection>
      <TitleBlock
        cells={[
          { label: "Sheet", value: "Q-02" },
          { label: "Drawing", value: questionNumber(id) },
        ]}
      />
    </>
  );
}

import Link from "next/link";

import type { QuestionDetailOut } from "@/client/types.gen";
import { Disclosure } from "@/components/disclosure";
import { Markdown } from "@/components/markdown";
import { number } from "@/lib/progress";
import { questionNumber, questionPath } from "@/lib/questions";
import {
  type Change,
  type Check,
  CHECKS,
  changedFields,
  type Edit,
  missingSomething,
  recordedPoints,
  type Report,
} from "@/lib/report";
import { dateLabel } from "@/lib/time";
import { cn } from "@/lib/utils";

// The record behind a question: the checks it went through when it was written, every
// correction since, and the model that wrote it.

const MONO = "font-mono text-[11px] leading-[1.4] tracking-[0.04em]";

type Outcome = "passed" | "failed" | "not checked";

const MARK: Record<Outcome, { mark: string; className: string }> = {
  passed: { mark: "✓", className: "" },
  failed: { mark: "✗", className: "text-thread" },
  "not checked": { mark: "–", className: "text-fg-3" },
};

const CHECK_NAME: Record<Check, string> = {
  quotes: "Quotes",
  answerable: "Answerable",
  trivia: "Not trivia",
  duplicate: "Not a duplicate",
};

function outcomeOf(check: Check, report: Report): Outcome {
  if (report.failed.includes(check)) return "failed";
  if (check === "answerable" && report.answerable === null) return "not checked";
  if (check === "trivia" && report.kind === null) return "not checked";
  if (check === "duplicate" && !report.duplicateChecked) return "not checked";
  return "passed";
}

function plural(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

function QuoteFindings({ report }: { report: Report }) {
  const total = report.quotes.length;
  if (total === 0) return <>No quotes were recorded.</>;
  const missed = report.quotes.filter((quote) => quote.problem !== null);
  return (
    <>
      {total - missed.length} of {plural(total, "quote")} found word for word in{" "}
      {total === 1 ? "its passage" : "their passages"}.
      {missed.length > 0 && (
        <ul className="mt-1.5 grid gap-1 text-fg-2">
          {missed.map((quote, i) => (
            <li key={i}>
              “<Markdown inline className="italic">{quote.quote}</Markdown>”: {quote.problem}
            </li>
          ))}
        </ul>
      )}
    </>
  );
}

function Finding({ check, report, outcome }: { check: Check; report: Report; outcome: Outcome }) {
  switch (check) {
    case "quotes":
      return <QuoteFindings report={report} />;
    case "answerable":
      if (report.answerable === null) return <>Not checked.</>;
      if (report.answerable) return <>A second model answered it from the passages alone.</>;
      return (
        <>
          A second model couldn&apos;t answer it from the passages
          {missingSomething(report.missing) ? <>. Missing: {report.missing}</> : null}.
        </>
      );
    case "trivia":
      if (report.kind === null) return <>Not checked.</>;
      return report.kind === "recall" ? (
        <>It reads as recalling a fact off the page, not explaining something.</>
      ) : (
        <>It asks to explain something, not to recall a fact.</>
      );
    case "duplicate": {
      if (!report.duplicateChecked) {
        return <>Not checked: the embedding model was away when it was written.</>;
      }
      if (report.nearest === null) return <>There was no other question to compare it with.</>;
      const nearest = (
        <Link href={questionPath(report.nearest.id)} className="thread-link">
          {questionNumber(report.nearest.id)}
        </Link>
      );
      const similarity = report.nearest.similarity?.toFixed(2);
      return outcome === "failed" ? (
        <>
          Too close to {nearest}
          {similarity && <>, at {similarity} similarity</>}.
        </>
      ) : (
        <>
          The nearest question is {nearest}
          {similarity && <>, at {similarity} similarity</>}.
        </>
      );
    }
  }
}

export function Checks({ report }: { report: Report }) {
  return (
    <>
      <ul className="max-w-[60rem]">
        {CHECKS.map((check) => {
          const outcome = outcomeOf(check, report);
          const { mark, className } = MARK[outcome];
          return (
            <li
              key={check}
              className="grid grid-cols-[20px_minmax(0,1fr)] items-baseline gap-x-3 gap-y-1 border-b border-dotted border-line-2 py-2.5 text-small sm:grid-cols-[20px_9rem_minmax(0,1fr)]"
            >
              <span
                role="img"
                aria-label={outcome}
                className={cn("text-center font-mono text-sm leading-none font-bold", className)}
              >
                {mark}
              </span>
              <span className="type-label">{CHECK_NAME[check]}</span>
              <div className="col-start-2 sm:col-start-auto">
                <Finding check={check} report={report} outcome={outcome} />
              </div>
            </li>
          );
        })}
      </ul>
      <p className={cn(MONO, "mt-3 text-fg-2")}>
        Checked by {report.checkerModel ?? "the local checker"}
        {report.promptVersion && ` · ${report.promptVersion}`}
        {report.agreement !== null &&
          ` · the reference answer and the checker's agree ${report.agreement.toFixed(2)} (recorded, not judged)`}
      </p>
      {report.checkerAnswer && (
        <Disclosure summary="The checker's answer, from the passages alone" className="mt-4">
          <Markdown className="text-small">{report.checkerAnswer}</Markdown>
        </Disclosure>
      )}
    </>
  );
}

const FIELD_TITLE: Record<string, string> = {
  text: "The question",
  reference_answer: "Reference answer",
  key_points: "Key points",
};

function Side({ label, field, value }: { label: string; field: string; value: unknown }) {
  return (
    <div className="min-w-0">
      <p className={cn(MONO, "mb-1.5 text-fg-3 uppercase")}>{label}</p>
      {field === "key_points" ? (
        <ol className="grid gap-1.5 text-small">
          {recordedPoints(value).map((point, i) => (
            <li key={i}>
              <span className={cn(MONO, "text-fg-2 uppercase")}>
                k{i + 1}
                {point.weight !== null && ` · weight ${point.weight}`}
              </span>{" "}
              <Markdown inline>{point.text}</Markdown>
            </li>
          ))}
        </ol>
      ) : typeof value === "string" ? (
        <Markdown className="text-small">{value}</Markdown>
      ) : (
        <p className="text-small">{JSON.stringify(value)}</p>
      )}
    </div>
  );
}

function Replaced({ changes }: { changes: Change[] }) {
  return (
    <div className="grid gap-6">
      {changes.map((change) => (
        <div key={change.field}>
          <h3 className="type-label text-fg-2">{FIELD_TITLE[change.field] ?? change.field}</h3>
          <div className="mt-2 grid grid-cols-1 gap-x-6 gap-y-3 md:grid-cols-2">
            <Side label="Before" field={change.field} value={change.from} />
            <Side label="After" field={change.field} value={change.to} />
          </div>
        </div>
      ))}
    </div>
  );
}

function statusWords(change: Change): string {
  if (change.to === "retired") return "retired";
  if (change.to === "accepted") return "put back in practice";
  return `status ${String(change.to)}`;
}

function EditEntry({ edit }: { edit: Edit }) {
  const status = edit.changes.find((change) => change.field === "status");
  const corrected = edit.changes.filter((change) => change.field !== "status");
  const what = [
    status && statusWords(status),
    corrected.length > 0 && `${changedFields(corrected)} corrected`,
    edit.quotes.length > 0 &&
      `${edit.quotes.filter((quote) => quote.problem === null).length} of ${plural(edit.quotes.length, "quote")} found in their passages`,
    edit.embedding && `embedding ${edit.embedding}`,
  ].filter(Boolean);
  return (
    <li className="border-l-2 border-line-2 pl-4">
      <p className={cn(MONO, "text-fg-2 uppercase")}>
        {edit.at ? dateLabel(edit.at.slice(0, 10)) : "undated"} · {what.join(" · ")}
      </p>
      {edit.reason && <p className="mt-1.5 max-w-[62ch] text-small">“{edit.reason}”</p>}
      {corrected.length > 0 && (
        <Disclosure summary="What it replaced" className="mt-3">
          <Replaced changes={corrected} />
        </Disclosure>
      )}
    </li>
  );
}

/** Every correction, the latest first. */
export function Edits({ edits }: { edits: Edit[] }) {
  if (edits.length === 0) {
    return (
      <p className="text-small text-fg-2">
        None yet. A correction is kept here with what it replaced and why.
      </p>
    );
  }
  return (
    <ol className="grid max-w-[60rem] gap-5">
      {edits
        .map((edit, i) => <EditEntry key={i} edit={edit} />)
        .reverse()}
    </ol>
  );
}

function tokens(usage: Record<string, unknown>): string | null {
  const { input_tokens: sent, output_tokens: written } = usage;
  if (typeof sent !== "number" || typeof written !== "number") return null;
  return `${number(sent + written)} tokens (${number(sent)} in, ${number(written)} out)`;
}

export function Origin({ question }: { question: QuestionDetailOut }) {
  const spent = tokens(question.usage);
  return (
    <p className={cn(MONO, "text-fg-2")}>
      {question.generator_model} · {question.prompt_version}
      {spent && ` · ${spent}`} · {dateLabel(question.created_at.slice(0, 10))}
    </p>
  );
}

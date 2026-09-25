"use client";

import { useQuery } from "@tanstack/react-query";

import { practiceNextOptions } from "@/client/@tanstack/react-query.gen";
import type { PracticeOut } from "@/client/types.gen";
import { Commands } from "@/components/commands";
import { Pips } from "@/components/hatching";
import { Markdown } from "@/components/markdown";
import { SheetSection, TitleBlock } from "@/components/sheet";
import { StepLabel, Thread, ThreadStep } from "@/components/thread";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { API_URL } from "@/lib/api";
import { ApiError } from "@/lib/api-errors";
import { questionNumber, styleLabel } from "@/lib/questions";

function sentence(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function Question({ pick }: { pick: PracticeOut }) {
  const { question } = pick;
  return (
    <ThreadStep>
      <StepLabel
        meta={
          <>
            {questionNumber(question.id)} · {question.topic ?? "no topic"} ·{" "}
            {styleLabel(question.style)} · difficulty{" "}
            <Pips value={question.difficulty} label="difficulty" />
          </>
        }
      >
        Question
      </StepLabel>
      <p className="mt-3 flex flex-wrap items-center gap-2 text-small text-fg-2">
        <Badge>Why this</Badge>
        {sentence(pick.why)}
      </p>
      <Markdown className="type-q mt-3 mb-2 max-w-[34em]">{question.text}</Markdown>
      <p className="font-mono text-[11px] leading-normal tracking-[0.03em] text-fg-2">
        Sources · {question.citations.join(" · ")}
      </p>
      {question.source_updated && (
        <p className="mt-2 font-mono text-[11px] leading-normal tracking-[0.03em] text-thread">
          A source was ingested again after this question was written.
        </p>
      )}
    </ThreadStep>
  );
}

function Problem({ error, retry }: { error: Error; retry: () => void }) {
  const status = error instanceof ApiError ? error.status : null;

  if (status === 404) {
    return (
      <ThreadStep>
        <StepLabel meta="none yet">Question</StepLabel>
        <p className="mt-3 mb-5 max-w-[62ch]">
          There is nothing to practise yet. Add study material, build the topic map and write
          questions from it:
        </p>
        <Commands
          title="Filling the labyrinth"
          lines={[
            { command: 'make ingest SRC="notes.pdf 1706.03762"' },
            { command: "make topics" },
            { command: "make generate N=20" },
          ]}
        />
      </ThreadStep>
    );
  }

  return (
    <ThreadStep>
      <StepLabel meta={status === null ? "API not reachable" : `API error ${status}`}>
        Question
      </StepLabel>
      <p className="mt-3 mb-4 max-w-[62ch]">
        {status === null ? (
          <>
            Can&apos;t reach the API at <code className="font-mono">{API_URL}</code>. Start it
            with <code className="font-mono">make api</code>, then try again.
          </>
        ) : (
          error.message
        )}
      </p>
      <Button variant="outline" size="sm" onClick={retry}>
        Try again
      </Button>
    </ThreadStep>
  );
}

export function NextQuestion({ children }: { children: React.ReactNode }) {
  const next = useQuery(practiceNextOptions());
  const pick = next.data;

  return (
    <>
      <SheetSection>
        {children}
        <Thread aria-live="polite" aria-busy={next.isPending}>
          {next.isPending && (
            <ThreadStep>
              <StepLabel meta="picking">Question</StepLabel>
              <p className="mt-3 text-fg-2">Finding the question to practise next…</p>
            </ThreadStep>
          )}
          {next.isError && <Problem error={next.error} retry={() => next.refetch()} />}
          {pick && <Question pick={pick} />}
          <ThreadStep end>
            <StepLabel meta="not open yet">Your answer</StepLabel>
          </ThreadStep>
        </Thread>
      </SheetSection>
      <TitleBlock
        cells={[
          { label: "Sheet", value: "P-01" },
          { label: "Drawing", value: "Practice" },
          { label: "Due today", value: pick ? pick.due_count : "–" },
          { label: "New", value: pick ? pick.new_count : "–" },
          {
            label: "Topic mastery",
            value: pick ? pick.topic_mastery.toFixed(2) : "–",
          },
        ]}
      />
    </>
  );
}

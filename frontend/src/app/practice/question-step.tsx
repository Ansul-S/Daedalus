"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import type { PracticeOut } from "@/client/types.gen";
import { Commands, FILL_THE_LABYRINTH } from "@/components/commands";
import { Pips } from "@/components/hatching";
import { Markdown } from "@/components/markdown";
import { StepLabel, ThreadStep } from "@/components/thread";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { API_URL } from "@/lib/api";
import { ApiError } from "@/lib/api-errors";
import { questionNumber, questionPath, styleLabel } from "@/lib/questions";

function sentence(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/** The question as the interviewer asks it, with why it came up and where it comes from.
 * `focus` moves the reader to it as it appears, when it replaces the one just answered. */
export function QuestionStep({ pick, focus = false }: { pick: PracticeOut; focus?: boolean }) {
  const { question } = pick;
  const step = useRef<HTMLDivElement>(null);
  const [focusOnArrival] = useState(focus);

  useEffect(() => {
    // The page is already on its way back to the top: don't scroll it anywhere else.
    if (focusOnArrival) step.current?.focus({ preventScroll: true });
  }, [focusOnArrival]);

  return (
    <ThreadStep>
      <div ref={step} tabIndex={-1} className="outline-none">
        <StepLabel
          as="h2"
          meta={
            <>
              <Link href={questionPath(question.id)} className="thread-link">
                {questionNumber(question.id)}
              </Link>{" "}
              · {question.topic ?? "no topic"} ·{" "}
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
      </div>
    </ThreadStep>
  );
}

export function QuestionLoading() {
  return (
    <ThreadStep>
      <StepLabel as="h2" meta="picking">
        Question
      </StepLabel>
      <p className="mt-3 text-fg-2">Finding the question to practise next…</p>
    </ThreadStep>
  );
}

/** No question to show: none written yet, or the API didn't answer. */
export function QuestionProblem({ error, retry }: { error: Error; retry: () => void }) {
  const status = error instanceof ApiError ? error.status : null;

  if (status === 404) {
    return (
      <ThreadStep>
        <StepLabel as="h2" meta="none yet">
          Question
        </StepLabel>
        <p className="mt-3 mb-5 max-w-[62ch]">
          There is nothing to practise yet. Add study material, build the topic map and write
          questions from it:
        </p>
        <Commands title="Filling the labyrinth" lines={FILL_THE_LABYRINTH} />
      </ThreadStep>
    );
  }

  return (
    <ThreadStep>
      <StepLabel as="h2" meta={status === null ? "API not reachable" : `API error ${status}`}>
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

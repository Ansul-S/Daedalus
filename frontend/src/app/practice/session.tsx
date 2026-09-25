"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { practiceNextOptions, practiceNextQueryKey } from "@/client/@tanstack/react-query.gen";
import { answerQuestion, gradeAgain } from "@/client/sdk.gen";
import type { AttemptOut, KeyPointGradeOut, PracticeOut } from "@/client/types.gen";
import { SheetSection, TitleBlock } from "@/components/sheet";
import { StepLabel, Thread, ThreadStep } from "@/components/thread";
import { ApiError } from "@/lib/api-errors";
import { clearDraft } from "@/lib/drafts";
import { isTyping } from "@/lib/keyboard";

import { AnswerStep, type Sent } from "./answer-step";
import { QuestionLoading, QuestionProblem, QuestionStep } from "./question-step";
import {
  GradingStep,
  NextStep,
  RequestFailedStep,
  UngradedStep,
  VerdictAhead,
  VerdictStep,
} from "./verdict";

// One question at a time: question → your answer → verdict → next, down Ariadne's thread.

type Send = { kind: "answer"; answer: string; seconds: number } | { kind: "regrade"; attemptId: number };

const NO_POINTS: KeyPointGradeOut[] = [];

/** One question, from the first word of the answer to its verdict. */
function Round({ pick, fresh, onNext }: { pick: PracticeOut; fresh: boolean; onNext: () => void }) {
  const { question } = pick;
  const [attempt, setAttempt] = useState<AttemptOut | null>(null);
  const focus = useRef<HTMLDivElement>(null);

  const grading = useMutation({
    mutationFn: async (send: Send): Promise<AttemptOut> => {
      if (send.kind === "regrade") {
        const { data } = await gradeAgain({
          path: { attempt_id: send.attemptId },
          throwOnError: true,
        });
        return data;
      }
      const { data } = await answerQuestion({
        path: { question_id: question.id },
        body: { answer: send.answer, seconds: send.seconds },
        throwOnError: true,
      });
      return data;
    },
    onSuccess: (result) => {
      setAttempt(result);
      if (result.grades.at(-1)?.status === "graded") clearDraft(question.id);
    },
    onError: (error, send) => {
      // A retired question won't be asked again: its draft has nowhere to go.
      if (send.kind === "answer" && error instanceof ApiError && error.status === 404) {
        clearDraft(question.id);
      }
    },
  });

  const grade = attempt?.grades.at(-1) ?? null;
  const phase = grading.isPending
    ? "grading"
    : grading.isError
      ? "error"
      : grade?.status === "graded"
        ? "graded"
        : grade
          ? "ungraded"
          : "writing";
  // Retired while it was being answered: nothing to do here but move on.
  const gone =
    phase === "error" &&
    attempt === null &&
    grading.error instanceof ApiError &&
    grading.error.status === 404;
  const canMoveOn = phase === "graded" || phase === "ungraded" || gone;

  const pending = grading.isPending ? grading.variables : undefined;
  const sent = useMemo<Sent | null>(() => {
    if (attempt) return { answer: attempt.answer, seconds: attempt.seconds };
    return pending?.kind === "answer" ? { answer: pending.answer, seconds: pending.seconds } : null;
  }, [attempt, pending]);
  const points = grade?.status === "graded" ? grade.key_points : NO_POINTS;

  function submit(answer: { answer: string; seconds: number }) {
    if (!grading.isPending) grading.mutate({ kind: "answer", ...answer });
  }

  function regrade() {
    if (attempt && !grading.isPending) grading.mutate({ kind: "regrade", attemptId: attempt.id });
  }

  // Take the reader to each new step: the wait, then the verdict.
  useEffect(() => {
    if (phase === "writing") return;
    focus.current?.focus({ preventScroll: true });
    focus.current?.scrollIntoView({ block: phase === "grading" ? "nearest" : "start" });
  }, [phase]);

  // N for the next question, once there is nothing left to do with this one.
  useEffect(() => {
    if (!canMoveOn) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "n" && event.key !== "N") return;
      if (event.metaKey || event.ctrlKey || event.altKey || event.repeat) return;
      if (event.defaultPrevented || isTyping(event.target)) return;
      event.preventDefault();
      onNext();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [canMoveOn, onNext]);

  return (
    <>
      <QuestionStep pick={pick} focus={fresh} />
      <AnswerStep
        questionId={question.id}
        passages={question.citations.length}
        sent={sent}
        points={points}
        onSubmit={submit}
      />
      {phase === "writing" && <VerdictAhead />}
      {phase === "grading" && <GradingStep focusRef={focus} />}
      {phase === "error" && grading.error && (
        <RequestFailedStep
          error={grading.error}
          answered={attempt !== null}
          onRegrade={regrade}
          onNext={onNext}
          focusRef={focus}
        />
      )}
      {phase === "graded" && grade && (
        <VerdictStep grade={grade} review={attempt?.review ?? null} focusRef={focus} />
      )}
      {phase === "ungraded" && grade && (
        <UngradedStep grade={grade} onRegrade={regrade} focusRef={focus} />
      )}
      {(phase === "graded" || phase === "ungraded") && <NextStep onNext={onNext} />}
    </>
  );
}

function AnswerAhead() {
  return (
    <ThreadStep end>
      <StepLabel as="h2" meta="not open yet">
        Your answer
      </StepLabel>
    </ThreadStep>
  );
}

export function PracticeSession({ children }: { children: React.ReactNode }) {
  const queryClient = useQueryClient();
  // Fetched when the page opens and when the reader moves on, never behind their back: a
  // refetch would swap the question under an answer being written, or a verdict being read.
  const next = useQuery({
    ...practiceNextOptions(),
    staleTime: Infinity,
    gcTime: 0,
    refetchOnReconnect: false,
  });
  const [movedOn, setMovedOn] = useState(false);
  const pick = next.data;

  const moveOn = useCallback(() => {
    setMovedOn(true);
    window.scrollTo({ top: 0 });
    void queryClient.resetQueries({ queryKey: practiceNextQueryKey() });
  }, [queryClient]);

  return (
    <>
      <SheetSection>
        {children}
        <Thread aria-busy={next.isFetching}>
          {pick ? (
            // Keyed by the fetch, so the same question served again starts afresh
            <Round
              key={`${pick.question.id}:${next.dataUpdatedAt}`}
              pick={pick}
              fresh={movedOn}
              onNext={moveOn}
            />
          ) : (
            <>
              {next.isError ? (
                <QuestionProblem error={next.error} retry={() => next.refetch()} />
              ) : (
                <QuestionLoading />
              )}
              <AnswerAhead />
            </>
          )}
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

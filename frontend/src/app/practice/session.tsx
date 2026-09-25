"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  practiceMapOptions,
  practiceMapQueryKey,
  practiceNextOptions,
  practiceNextQueryKey,
  practiceProgressOptions,
  practiceProgressQueryKey,
  practiceStatsQueryKey,
} from "@/client/@tanstack/react-query.gen";
import { answerQuestion, gradeAgain } from "@/client/sdk.gen";
import type { AttemptOut, KeyPointGradeOut, PracticeOut } from "@/client/types.gen";
import { SheetSection, TitleBlock } from "@/components/sheet";
import { StepLabel, Thread, ThreadStep } from "@/components/thread";
import { ApiError } from "@/lib/api-errors";
import { clearDraft } from "@/lib/drafts";
import { useInterviewMode } from "@/lib/interview";
import { isTyping } from "@/lib/keyboard";
import { days, number } from "@/lib/progress";

import { AnswerStep, type Sent } from "./answer-step";
import { announce } from "./earned";
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

type Send =
  | { kind: "answer"; answer: string; seconds: number; timeLimit: number | null }
  | { kind: "regrade"; attemptId: number };

const NO_POINTS: KeyPointGradeOut[] = [];

/** One question, from the first word of the answer to its verdict. `onReviewed` is told when
 * the answer's grade has rescheduled the question and earned its XP. */
function Round({
  pick,
  fresh,
  interview,
  onReviewed,
  onNext,
}: {
  pick: PracticeOut;
  fresh: boolean;
  interview: boolean;
  onReviewed: () => void;
  onNext: () => void;
}) {
  const { question } = pick;
  const queryClient = useQueryClient();
  const [attempt, setAttempt] = useState<AttemptOut | null>(null);
  const focus = useRef<HTMLDivElement>(null);
  const rewarded = useRef(false);

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
        body: { answer: send.answer, seconds: send.seconds, time_limit: send.timeLimit },
        throwOnError: true,
      });
      return data;
    },
    onSuccess: (result) => {
      setAttempt(result);
      if (result.grades.at(-1)?.status === "graded") clearDraft(question.id);
      // What the first successful grade earned is announced once, whatever comes back later
      if (result.earned && !rewarded.current) {
        rewarded.current = true;
        onReviewed();
        announce(result.earned);
        for (const queryKey of [
          practiceProgressQueryKey(),
          practiceMapQueryKey(),
          practiceStatsQueryKey(),
        ]) {
          void queryClient.invalidateQueries({ queryKey });
        }
      }
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
    if (attempt) {
      return { answer: attempt.answer, seconds: attempt.seconds, timeLimit: attempt.time_limit };
    }
    return pending?.kind === "answer"
      ? { answer: pending.answer, seconds: pending.seconds, timeLimit: pending.timeLimit }
      : null;
  }, [attempt, pending]);
  const points = grade?.status === "graded" ? grade.key_points : NO_POINTS;

  function submit(answer: { answer: string; seconds: number; timeLimit: number | null }) {
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
        interview={interview}
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
        <VerdictStep
          grade={grade}
          review={attempt?.review ?? null}
          earned={attempt?.earned ?? null}
          focusRef={focus}
        />
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
  const interview = useInterviewMode();
  const progress = useQuery(practiceProgressOptions());
  // The round whose answer has been reviewed: it is no longer due, or no longer new
  const round = pick ? `${pick.question.id}:${next.dataUpdatedAt}` : null;
  const [reviewedRound, setReviewedRound] = useState<string | null>(null);
  const reviewed = round !== null && reviewedRound === round;
  // Its topic's mastery moves with the grade: read again from the map once it is in
  const map = useQuery({ ...practiceMapOptions(), enabled: reviewed });
  const room = map.data?.rooms.find((candidate) => candidate.topic_id === pick?.question.topic_id);
  const found = progress.data;

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
              key={round}
              pick={pick}
              fresh={movedOn}
              interview={interview}
              onReviewed={() => setReviewedRound(round)}
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
          {
            label: "Due today",
            value: pick ? pick.due_count - (reviewed && pick.reason === "due" ? 1 : 0) : "–",
          },
          {
            label: "New",
            value: pick ? pick.new_count - (reviewed && pick.reason === "new" ? 1 : 0) : "–",
          },
          {
            label: "Topic mastery",
            value: pick ? (reviewed && room ? room.mastery : pick.topic_mastery).toFixed(2) : "–",
          },
          { label: "Thread", value: found ? days(found.streak.days) : "–" },
          { label: "Level", value: found ? `${found.level.number} · ${found.level.name}` : "–" },
          { label: "XP", value: found ? number(found.xp) : "–" },
        ]}
      />
    </>
  );
}

"use client";

import { useQueries, useQuery } from "@tanstack/react-query";

import { listAttemptsOptions, listQuestionsOptions } from "@/client/@tanstack/react-query.gen";
import type { AttemptOut, RoomOut } from "@/client/types.gen";
import { Pips } from "@/components/hatching";
import { Markdown } from "@/components/markdown";
import { Button } from "@/components/ui/button";
import { questionNumber, styleLabel } from "@/lib/questions";
import { dayLabel } from "@/lib/time";
import { cn } from "@/lib/utils";

// A room opened on the map: its questions, with the latest score of each and when it comes
// back. The question bank will link each one to its page.

const MONO = "font-mono text-[11px] leading-[1.35] tracking-[0.04em]";

/** Where a question stands after its latest review: that answer's score and the day it is
 * due again. Null when it has never been graded. */
function standing(attempts: AttemptOut[]): { score: number | null; due: string } | null {
  // Newest first; only an answer's first successful grade reviews it
  const reviewed = attempts.find((attempt) => attempt.review !== null);
  if (!reviewed?.review) return null;
  const grade = reviewed.grades.find((found) => found.status === "graded");
  return { score: grade?.score ?? null, due: reviewed.review.due };
}

function Standing({
  attempts,
  today,
}: {
  attempts: { data?: AttemptOut[]; isError: boolean };
  today: string;
}) {
  if (attempts.isError) return <span className="text-fg-3">not known</span>;
  if (!attempts.data) return <span className="text-fg-3">…</span>;
  const found = standing(attempts.data);
  if (!found) return <span className="text-fg-3">not answered yet</span>;
  const due = found.due <= today;
  return (
    <>
      {found.score !== null && <>last {found.score.toFixed(2)} · </>}
      <span className={cn(due && "text-thread")}>{due ? "due now" : `next ${dayLabel(found.due)}`}</span>
    </>
  );
}

export function RoomPanel({
  room,
  lair,
  today,
  onClose,
}: {
  room: RoomOut;
  lair: boolean;
  today: string;
  onClose: () => void;
}) {
  const questions = useQuery(
    listQuestionsOptions({ query: { topic_id: room.topic_id, status: "accepted", limit: 100 } }),
  );
  const found = [...(questions.data?.results ?? [])].sort((a, b) => a.id - b.id);
  // Fetched each time a room opens, so an answer given since shows at once
  const attempts = useQueries({
    queries: found.map((question) => ({
      ...listAttemptsOptions({ path: { question_id: question.id } }),
      staleTime: 0,
    })),
  });

  return (
    <div className="border border-line-2">
      <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2 border-b border-line-2 px-4 py-3">
        <div className="min-w-0">
          <h3 className="font-display text-[1.3rem] leading-none font-bold tracking-[0.04em] uppercase">
            {room.name}
          </h3>
          <p className={cn(MONO, "mt-1.5 text-fg-2 uppercase")}>
            {room.questions} question{room.questions === 1 ? "" : "s"} · {room.practised} practised ·
            mastery {room.mastery.toFixed(2)}
            {room.due > 0 && <span className="text-thread"> · {room.due} due</span>}
            {lair && " · the Minotaur's room"}
          </p>
        </div>
        <Button variant="ghost" size="sm" onClick={onClose} aria-keyshortcuts="Escape">
          Close
        </Button>
      </div>
      {questions.isError ? (
        <p className="px-4 py-3 text-small text-fg-2">
          The questions couldn&apos;t be loaded:{" "}
          {questions.error instanceof Error ? questions.error.message.replace(/\.$/, "") : "the API refused"}.
        </p>
      ) : !questions.data ? (
        <p className="px-4 py-3 text-small text-fg-2">Opening the room…</p>
      ) : (
        <ul>
          {found.map((question, i) => (
            <li
              key={question.id}
              className="grid grid-cols-1 gap-x-6 gap-y-1 border-line-2 px-4 py-3 not-first:border-t not-first:border-dotted sm:grid-cols-[minmax(0,1fr)_auto]"
            >
              <p className={cn(MONO, "text-fg-2 uppercase")}>
                {questionNumber(question.id)} · {styleLabel(question.style)} · difficulty{" "}
                <Pips value={question.difficulty} label="difficulty" />
              </p>
              <p className={cn(MONO, "text-fg-2 sm:row-start-1 sm:col-start-2 sm:text-right")}>
                <Standing attempts={attempts[i] ?? { isError: false }} today={today} />
              </p>
              <Markdown inline className="text-small sm:col-span-2">
                {question.text}
              </Markdown>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

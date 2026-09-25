"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import type { KeyPointGradeOut } from "@/client/types.gen";
import { DimensionMini, DimensionTimer } from "@/components/dimension-timer";
import { Markdown } from "@/components/markdown";
import { ThreadStep } from "@/components/thread";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { loadDraft, saveDraft } from "@/lib/drafts";
import {
  type Highlight,
  keyPointState,
  MAX_ANSWER_CHARS,
  MAX_SECONDS,
  splitAnswer,
  traceQuotes,
} from "@/lib/grades";
import { INTERVIEW_LIMIT, setInterviewMode } from "@/lib/interview";
import { isSubmitKey, useSubmitKeys } from "@/lib/keyboard";
import { useStopwatch } from "@/lib/stopwatch";
import { clock } from "@/lib/time";
import { cn } from "@/lib/utils";

/** An answer as it went to the grader, with the time it took (null when not recorded) and
 * the interview limit it was given, if any. */
export type Sent = { answer: string; seconds: number | null; timeLimit: number | null };

/** Interview mode, switched on and off beside the answer; it holds for the next questions. */
function InterviewSwitch({ on }: { on: boolean }) {
  return (
    <button
      type="button"
      aria-pressed={on}
      onClick={() => setInterviewMode(!on)}
      title={`Answer against a ${clock(INTERVIEW_LIMIT)} clock, as in an interview. Inside it earns 5 XP more.`}
      className="type-label inline-flex cursor-pointer items-center gap-2 border border-line-2 px-[9px] py-[5px] whitespace-nowrap hover:bg-surface aria-pressed:bg-fg aria-pressed:text-ground"
    >
      <i aria-hidden className={cn("size-2.5 border border-current", on ? "hatch-5" : "hatch-0")} />
      Interview · {clock(INTERVIEW_LIMIT)}
    </button>
  );
}

const MARK = "underline decoration-[1.5px] underline-offset-[5px]";
const COVERED = "decoration-fg";
const PARTIAL = "decoration-fg-2 decoration-dashed";

/** The answer with the words that earned each key point underlined: solid where the point is
 * covered, dashed where it is partly covered. */
function TracedAnswer({ answer, highlights }: { answer: string; highlights: Highlight[] }) {
  return (
    <div className="max-w-[48rem] border border-line-2 bg-surface px-4 py-3.5 text-base leading-[1.62] break-words whitespace-pre-wrap">
      {splitAnswer(answer, highlights).map(({ text, point }, i) =>
        point ? (
          <mark
            key={i}
            title={`Key point: ${point.text}`}
            className={cn(
              "bg-[color-mix(in_oklab,var(--fg)_7%,transparent)] text-inherit",
              MARK,
              keyPointState(point.status) === "partial" ? PARTIAL : COVERED,
            )}
          >
            {text}
          </mark>
        ) : (
          text
        ),
      )}
    </div>
  );
}

/** Your answer: written and previewed here, timed, and kept as a draft until it is graded.
 * In `interview` mode the clock counts down from three minutes and runs on into overtime.
 * Once `sent`, it stays as it went to the grader. */
export function AnswerStep({
  questionId,
  passages,
  interview,
  sent,
  points,
  onSubmit,
}: {
  questionId: number;
  passages: number;
  interview: boolean;
  sent: Sent | null;
  /** The latest grade's key points, to trace the words that earned them. */
  points: KeyPointGradeOut[];
  onSubmit: (sent: { answer: string; seconds: number; timeLimit: number | null }) => void;
}) {
  const [draft] = useState(() => loadDraft(questionId));
  const [answer, setAnswer] = useState(draft?.answer ?? "");
  const [view, setView] = useState("write");
  const latest = useRef(answer);
  const locked = sent !== null;
  const { seconds, read } = useStopwatch(draft?.seconds ?? 0, !locked, interview);
  const keys = useSubmitKeys();

  // The time is kept as well when the reader leaves, so a reload carries on from there; so is
  // leaving for another page of the app, which unmounts the answer without a pagehide.
  useEffect(() => {
    if (locked) return;
    const keep = () => saveDraft(questionId, { answer: latest.current, seconds: read() });
    const onVisibility = () => {
      if (document.hidden) keep();
    };
    document.addEventListener("visibilitychange", onVisibility);
    window.addEventListener("pagehide", keep);
    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("pagehide", keep);
      keep();
    };
  }, [questionId, locked, read]);

  function change(value: string) {
    latest.current = value;
    setAnswer(value);
    saveDraft(questionId, { answer: value, seconds: read() });
  }

  function submit() {
    if (locked || !answer.trim()) return;
    onSubmit({
      answer,
      seconds: Math.min(Math.round(read()), MAX_SECONDS),
      timeLimit: interview ? INTERVIEW_LIMIT : null,
    });
  }

  const text = sent?.answer ?? answer;
  const preview = useMemo(() => <Markdown>{text}</Markdown>, [text]);
  const highlights = useMemo(() => (sent ? traceQuotes(sent.answer, points) : []), [sent, points]);

  return (
    <ThreadStep>
      <Tabs
        value={view}
        onValueChange={setView}
        className="gap-3"
        onKeyDown={(event) => {
          if (!isSubmitKey(event)) return;
          event.preventDefault();
          submit();
        }}
      >
        <div className="type-label flex max-w-[48rem] flex-wrap items-center gap-x-3.5 gap-y-2">
          <h2>Your answer</h2>
          <TabsList aria-label="Show the answer as">
            <TabsTrigger value="write">{locked ? "Text" : "Write"}</TabsTrigger>
            <TabsTrigger value="preview">Preview</TabsTrigger>
          </TabsList>
          {!locked && <InterviewSwitch on={interview} />}
          {sent
            ? sent.seconds !== null && (
                <DimensionMini
                  className="ml-auto"
                  seconds={sent.seconds}
                  limit={sent.timeLimit ?? undefined}
                />
              )
            : !interview && <DimensionMini className="ml-auto" seconds={seconds} />}
        </div>
        {!locked && interview && (
          <DimensionTimer elapsed={seconds} limit={INTERVIEW_LIMIT} className="max-w-[48rem]" />
        )}
        {/* While writing, the textarea is the stop in the tab order, not its panel */}
        <TabsContent value="write" tabIndex={sent ? 0 : -1}>
          {sent ? (
            <TracedAnswer answer={sent.answer} highlights={highlights} />
          ) : (
            <Textarea
              aria-label="Your answer"
              value={answer}
              onChange={(event) => change(event.target.value)}
              maxLength={MAX_ANSWER_CHARS}
              placeholder="Answer as you would out loud in an interview. Markdown and $maths$ work."
              className="min-h-44 max-w-[48rem]"
            />
          )}
        </TabsContent>
        <TabsContent value="preview">
          <div className="min-h-44 max-w-[48rem] border border-line-2 bg-surface px-4 py-3.5">
            {text.trim() ? preview : <p className="text-fg-3">Nothing to preview yet.</p>}
          </div>
        </TabsContent>
      </Tabs>
      {sent ? (
        highlights.length > 0 && (
          <p className="mt-2.5 font-mono text-[11px] leading-normal tracking-[0.03em] text-fg-2">
            Your words the grader credited: <span className={cn(MARK, COVERED)}>covered</span> ·{" "}
            <span className={cn(MARK, PARTIAL)}>partly covered</span>
          </p>
        )
      ) : (
        <div className="mt-3 flex max-w-[48rem] flex-wrap items-center gap-x-4 gap-y-2.5">
          <Button size="sm" onClick={submit} disabled={!answer.trim()} aria-keyshortcuts={keys.aria}>
            Submit answer <kbd aria-hidden>{keys.label}</kbd>
          </Button>
          <span className="font-mono text-[11px] leading-[1.3] tracking-[0.05em] text-fg-2">
            Graded against the {passages === 1 ? "passage" : `${passages} passages`} above
          </span>
          <span className="ml-auto font-mono text-[11px] leading-[1.3] tracking-[0.05em] text-fg-2 tabular-nums">
            {answer.length.toLocaleString("en")} / {MAX_ANSWER_CHARS.toLocaleString("en")}
          </span>
        </div>
      )}
    </ThreadStep>
  );
}

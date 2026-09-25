"use client";

import type {
  ClaimOut,
  EarnedOut,
  GradeOut,
  KeyPointGradeOut,
  QuestionOut,
  ReviewOut,
} from "@/client/types.gen";
import { Disclosure } from "@/components/disclosure";
import {
  KeyPointMark,
  Pips,
  RatingChip,
  type Verdict,
  VerdictMark,
} from "@/components/hatching";
import { Markdown } from "@/components/markdown";
import { Rate } from "@/components/rating";
import { StepLabel, ThreadStep } from "@/components/thread";
import { Button } from "@/components/ui/button";
import { API_URL } from "@/lib/api";
import { ApiError } from "@/lib/api-errors";
import {
  comeBack,
  gradedBy,
  keyPointState,
  RATING_LABEL,
  scoreFormula,
  verdictOf,
} from "@/lib/grades";
import { useElapsed } from "@/lib/stopwatch";
import { cn } from "@/lib/utils";

import { Earned } from "./earned";

// The verdict on an answer, as drawn on the pattern book's practice sheet: the score and what
// it earns, how the score was reached, the XP it brought, each key point, each claim against
// its passage, and the grader's notes; then your own rating of the question and of the grade.

type FocusRef = React.Ref<HTMLDivElement>;

// Where the reader is taken when a step appears, clear of the sticky header
const FOCUS = "scroll-mt-28 outline-none";
const SUB = "type-label mt-[22px] mb-2 text-fg-2";
const MONO = "font-mono text-[11px] leading-[1.4] tracking-[0.03em]";

/** Before the answer is sent, the thread ends at the verdict still to come. */
export function VerdictAhead() {
  return (
    <ThreadStep end>
      <StepLabel as="h2" meta="once you submit">
        Verdict
      </StepLabel>
    </ThreadStep>
  );
}

export function GradingStep({ focusRef }: { focusRef: FocusRef }) {
  const seconds = useElapsed();
  return (
    <ThreadStep end>
      <div ref={focusRef} tabIndex={-1} className={FOCUS}>
        <StepLabel as="h2" meta={`grading · ${seconds} s`}>
          Verdict
        </StepLabel>
        <p className="mt-3 text-fg-2">The grader is reading your answer against the sources…</p>
        {seconds >= 10 && (
          <p className="mt-2 max-w-[62ch] text-small text-fg-2">
            Still reading. Groq&apos;s free tier makes a second answer within the same minute wait
            its turn, and without Groq the local model takes over a minute.
          </p>
        )}
      </div>
    </ThreadStep>
  );
}

function ScoreRow({ score, review }: { score: number; review: ReviewOut | null }) {
  return (
    <div className="mt-3 flex flex-wrap items-baseline gap-x-[22px] gap-y-2.5">
      <span className="type-figure text-[clamp(3.6rem,7vw,5.4rem)] leading-[0.8]">
        <span className="sr-only">Score </span>
        {score.toFixed(2)}
      </span>
      {review && (
        <>
          <span className="inline-flex items-center gap-2 font-display text-[1.3rem] leading-none font-bold tracking-[0.05em] uppercase">
            <span aria-hidden className="inline-flex">
              <RatingChip rating={review.rating} />
            </span>
            {RATING_LABEL[review.rating]}
          </span>
          <span className="font-mono text-xs leading-[1.4] tracking-[0.04em] text-fg-2">
            {comeBack(review)}
          </span>
        </>
      )}
    </div>
  );
}

function KeyPoints({ points }: { points: KeyPointGradeOut[] }) {
  return (
    <>
      <h3 className={SUB}>Key points</h3>
      <ul className="max-w-[60rem]">
        {points.map((point) => {
          const state = keyPointState(point.status);
          return (
            <li
              key={point.id}
              className="grid grid-cols-[20px_minmax(0,1fr)] items-baseline gap-x-3 gap-y-1 border-b border-dotted border-line-2 py-[9px] text-small sm:grid-cols-[20px_minmax(0,1fr)_auto]"
            >
              <KeyPointMark state={state} className="translate-y-0.5" />
              <div className={cn(state === "missing" && "text-fg-2")}>
                <Markdown inline>{point.text}</Markdown>
                {state !== "missing" && point.answer_quote && (
                  <p className="mt-1 text-fg-2">
                    <span className="sr-only">Your words: </span>“
                    <Markdown inline className="italic">
                      {point.answer_quote}
                    </Markdown>
                    ”
                    {point.quote_found === false && (
                      <span className={cn(MONO, "ml-2")}>not word for word</span>
                    )}
                  </p>
                )}
              </div>
              <span className="col-start-2 font-mono text-[11px] leading-[1.3] tracking-[0.06em] whitespace-nowrap text-fg-2 uppercase sm:col-start-auto">
                {state} · weight {point.weight}
              </span>
            </li>
          );
        })}
      </ul>
    </>
  );
}

function ClaimSource({ claim, verdict }: { claim: ClaimOut; verdict: Verdict }) {
  const box = cn(
    MONO,
    "col-start-2 max-w-[24rem] justify-self-start border px-2 py-1 md:col-start-auto md:justify-self-auto",
  );
  if (verdict === "unverified") {
    return (
      <span className={cn(box, "border-dashed border-line-2 text-fg-2")}>
        Not in these passages · unverified, no penalty
      </span>
    );
  }
  const source =
    claim.citation &&
    (claim.link ? (
      <a
        href={claim.link}
        target="_blank"
        rel="noreferrer"
        className="underline decoration-line-2 underline-offset-2 hover:decoration-current"
      >
        {claim.citation}
      </a>
    ) : (
      claim.citation
    ));
  if (verdict === "contradicted") {
    return (
      <span className={cn(box, "border-thread text-thread")}>
        Contradicts {source || "the passages"}: <Markdown inline>{claim.why}</Markdown>
      </span>
    );
  }
  return <span className={cn(box, "border-line-2")}>{source || claim.why}</span>;
}

function Claims({ claims }: { claims: ClaimOut[] }) {
  return (
    <>
      <h3 className={SUB}>Claims</h3>
      <ul className="max-w-[60rem]">
        {claims.map((claim, i) => {
          const verdict = verdictOf(claim.verdict);
          return (
            <li
              key={i}
              className="grid grid-cols-[20px_minmax(0,1fr)] items-center gap-x-3 gap-y-1.5 border-b border-dotted border-line-2 py-[9px] text-small md:grid-cols-[20px_minmax(0,max-content)_minmax(24px,1fr)_auto]"
            >
              <VerdictMark verdict={verdict} />
              <q
                className={cn(
                  verdict === "contradicted" && "line-through decoration-thread decoration-[1.5px]",
                  verdict === "unverified" &&
                    "underline decoration-fg-3 decoration-dotted underline-offset-[3px]",
                )}
              >
                <Markdown inline>{claim.claim}</Markdown>
              </q>
              <span aria-hidden className="hidden h-px bg-line-2 md:block" />
              <ClaimSource claim={claim} verdict={verdict} />
            </li>
          );
        })}
      </ul>
    </>
  );
}

function Note({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="min-w-0">
      <h3 className={SUB}>{title}</h3>
      <div className="text-small">{children}</div>
    </div>
  );
}

function Notes({ grade }: { grade: GradeOut }) {
  const lists = [
    ["Strength", grade.strengths],
    ["Gap", grade.gaps],
    ["Error", grade.errors],
  ] as const;
  return (
    <div className="grid max-w-[60rem] grid-cols-1 gap-x-7 md:grid-cols-3">
      {grade.clarity !== null && (
        <Note title="Clarity">
          <Pips value={grade.clarity} label="clarity" />{" "}
          <span aria-hidden>{grade.clarity} of 5</span>
        </Note>
      )}
      {lists.map(
        ([name, items]) =>
          items.length > 0 && (
            <Note key={name} title={items.length === 1 ? name : `${name}s`}>
              <ul className="grid gap-1.5">
                {items.map((item, i) => (
                  <li key={i}>
                    <Markdown inline>{item}</Markdown>
                  </li>
                ))}
              </ul>
            </Note>
          ),
      )}
      {grade.follow_up && (
        <Note title="Follow-up">
          <Markdown>{grade.follow_up}</Markdown>
        </Note>
      )}
    </div>
  );
}

function ModelAnswer({ text }: { text: string }) {
  return (
    <Disclosure summary="Model answer, drawn from the passages" className="mt-7">
      <Markdown>{text}</Markdown>
    </Disclosure>
  );
}

/** Was the question worth asking, and was the grade fair? Kept as evaluation data. */
function Ratings({ question, grade }: { question: QuestionOut; grade: GradeOut }) {
  return (
    <>
      <h3 className={SUB}>Your ratings</h3>
      <div className="flex max-w-[60rem] flex-wrap items-start gap-x-10 gap-y-4">
        <Rate rated="question" id={question.id} initial={question.rating} />
        <Rate key={grade.id} rated="grade" id={grade.id} initial={grade.rating} />
      </div>
    </>
  );
}

export function VerdictStep({
  question,
  grade,
  review,
  earned,
  focusRef,
}: {
  question: QuestionOut;
  grade: GradeOut;
  review: ReviewOut | null;
  earned: EarnedOut | null;
  focusRef: FocusRef;
}) {
  const formula = scoreFormula(grade);
  const by = gradedBy(grade);
  return (
    <ThreadStep>
      <div ref={focusRef} tabIndex={-1} className={FOCUS}>
        <StepLabel as="h2">Verdict</StepLabel>
        {grade.score !== null && <ScoreRow score={grade.score} review={review} />}
        {formula && <p className="mt-3 font-mono text-xs leading-normal text-fg-2">{formula}</p>}
        {earned && <Earned earned={earned} />}
      </div>
      <KeyPoints points={grade.key_points} />
      {grade.claims.length > 0 && <Claims claims={grade.claims} />}
      <Notes grade={grade} />
      {grade.improved_answer && <ModelAnswer text={grade.improved_answer} />}
      {by && <p className={cn(MONO, "mt-6 text-fg-2")}>{by}</p>}
      <Ratings question={question} grade={grade} />
    </ThreadStep>
  );
}

/** An answer no model could grade: it is kept, and can be graded again. */
export function UngradedStep({
  grade,
  onRegrade,
  focusRef,
}: {
  grade: GradeOut;
  onRegrade: () => void;
  focusRef: FocusRef;
}) {
  return (
    <ThreadStep>
      <div ref={focusRef} tabIndex={-1} className={FOCUS}>
        <StepLabel as="h2" meta="not graded">
          Verdict
        </StepLabel>
        <p className="mt-3 max-w-[62ch]">
          Your answer is saved, but no model could grade it. Grading it again tries each model in
          turn.
        </p>
      </div>
      <p
        className={cn(
          MONO,
          "mt-3 max-w-[62ch] border border-l-[3px] border-line-2 border-l-thread px-[18px] py-3 break-words text-fg-2",
        )}
      >
        {grade.error ?? "No reason was given."}
      </p>
      <Button size="sm" className="mt-4" onClick={onRegrade}>
        Grade again
      </Button>
    </ThreadStep>
  );
}

/** The API didn't take the answer, or didn't answer at all. */
export function RequestFailedStep({
  error,
  answered,
  onRegrade,
  onNext,
  focusRef,
}: {
  error: Error;
  /** Whether the answer is already saved, so it can be graded again as it stands. */
  answered: boolean;
  onRegrade: () => void;
  onNext: () => void;
  focusRef: FocusRef;
}) {
  const status = error instanceof ApiError ? error.status : null;
  const gone = status === 404 && !answered;
  const again = answered ? "grade it again." : "submit again: your answer is kept above.";

  return (
    <ThreadStep end>
      <div ref={focusRef} tabIndex={-1} className={FOCUS}>
        <StepLabel as="h2" meta={status === null ? "API not reachable" : `API error ${status}`}>
          Verdict
        </StepLabel>
        <p className="mt-3 max-w-[62ch]">
          {status === null ? (
            <>
              Can&apos;t reach the API at <code className="font-mono">{API_URL}</code>. Start it
              with <code className="font-mono">make api</code>, then {again}
            </>
          ) : gone ? (
            "This question is no longer in the library: it was retired while you were answering."
          ) : (
            `${error.message.replace(/\.$/, "")}. You can ${again}`
          )}
        </p>
      </div>
      {answered ? (
        <Button size="sm" className="mt-4" onClick={onRegrade}>
          Grade again
        </Button>
      ) : (
        gone && (
          <Button
            variant="outline"
            size="sm"
            className="mt-4"
            onClick={onNext}
            aria-keyshortcuts="N"
          >
            Next question <kbd aria-hidden>N</kbd>
          </Button>
        )
      )}
    </ThreadStep>
  );
}

export function NextStep({ onNext }: { onNext: () => void }) {
  return (
    <ThreadStep end>
      <Button variant="outline" size="sm" onClick={onNext} aria-keyshortcuts="N">
        Next question <kbd aria-hidden>N</kbd>
      </Button>
    </ThreadStep>
  );
}

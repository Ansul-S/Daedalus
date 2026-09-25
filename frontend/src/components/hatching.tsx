import { cn } from "@/lib/utils";

// Engravers show tone with hatching; Daedalus uses the same marks for status, so no state
// rests on colour alone and every one still reads in black and white.

const MASTERY = ["hatch-0", "hatch-1", "hatch-2", "hatch-3", "hatch-4", "hatch-5"] as const;

/** A room's mastery in six steps, 0 (untouched) to 5 (solid). */
export function MasteryHatch({ level, className }: { level: number; className?: string }) {
  const step = Math.min(5, Math.max(0, Math.round(level)));
  return (
    <i
      role="img"
      aria-label={`mastery ${step} of 5`}
      className={cn("block aspect-square border-[1.5px] border-current", MASTERY[step], className)}
    />
  );
}

export type KeyPointState = "covered" | "partial" | "missing";

const KEY_POINT: Record<KeyPointState, string> = {
  covered: "hatch-5",
  partial: "hatch-partial",
  missing: "border-fg-3",
};

/** Covered counts in full, partial counts half, missing counts nothing. */
export function KeyPointMark({ state, className }: { state: KeyPointState; className?: string }) {
  return (
    <i
      role="img"
      aria-label={state}
      className={cn(
        "inline-block size-3.5 shrink-0 border-[1.5px] border-current",
        KEY_POINT[state],
        className,
      )}
    />
  );
}

export type Rating = "again" | "hard" | "good" | "easy";

const RATING: Record<Rating, string> = {
  again: "hatch-0",
  hard: "hatch-hard",
  good: "hatch-good",
  easy: "hatch-5",
};

/** The four review ratings, darker as the answer gets stronger. */
export function RatingChip({ rating, className }: { rating: Rating; className?: string }) {
  return (
    <i
      role="img"
      aria-label={rating}
      className={cn(
        "inline-block size-[18px] shrink-0 border-[1.5px] border-current",
        RATING[rating],
        className,
      )}
    />
  );
}

export type Verdict = "supported" | "contradicted" | "unverified";

const VERDICT: Record<Verdict, { mark: string; className: string }> = {
  supported: { mark: "✓", className: "" },
  contradicted: { mark: "✗", className: "text-thread" },
  unverified: { mark: "?", className: "text-fg-3" },
};

/** A claim checked against the question's passages. */
export function VerdictMark({ verdict, className }: { verdict: Verdict; className?: string }) {
  const { mark, className: tone } = VERDICT[verdict];
  return (
    <span
      role="img"
      aria-label={verdict}
      className={cn("text-center font-mono text-sm leading-none font-bold", tone, className)}
    >
      {mark}
    </span>
  );
}

/** Difficulty as five pips: ▮▮▮▯▯. */
export function Pips({ value, of = 5, label }: { value: number; of?: number; label: string }) {
  const filled = Math.min(of, Math.max(0, Math.round(value)));
  return (
    <span role="img" aria-label={`${label} ${filled} of ${of}`} className="tracking-[0.08em]">
      {"▮".repeat(filled) + "▯".repeat(of - filled)}
    </span>
  );
}

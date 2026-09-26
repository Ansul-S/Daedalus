import type { JobState } from "@/lib/library";
import { cn } from "@/lib/utils";

// Work in the queue, marked the way hatching marks everything else, so no state rests on
// colour: an empty dashed box waits, a hatched one is under way, a solid one is done and a
// struck-through one failed. Hatched but dashed, it was stopped part way.

export const JOB_LABEL: Record<JobState, string> = {
  queued: "Queued",
  running: "Running",
  interrupted: "Stopped part way",
  done: "Done",
  failed: "Failed",
};

const MARK: Record<JobState, string> = {
  queued: "border-dashed",
  running: "hatch-partial",
  interrupted: "border-dashed hatch-partial",
  done: "hatch-5",
  failed: "hatch-cross text-thread",
};

/** A job's state as a mark; one under way sends out a ring. */
export function JobMark({ state, className }: { state: JobState; className?: string }) {
  return (
    <span className={cn("relative inline-grid size-3.5 shrink-0", className)}>
      {state === "running" && (
        <span
          aria-hidden
          className="absolute inset-0 animate-due-pulse border-[1.5px] border-current opacity-0"
        />
      )}
      <i
        role="img"
        aria-label={JOB_LABEL[state].toLowerCase()}
        className={cn("block size-3.5 border-[1.5px] border-current", MARK[state])}
      />
    </span>
  );
}

/** The mark and the state in words, then what the job is doing or how it ended; a long line
 * wraps under the words, clear of the mark. `label` names a finished state in the job's own
 * terms, such as Ready or Built. */
export function JobLine({
  state,
  label,
  children,
  className,
}: {
  state: JobState;
  label?: string;
  children?: React.ReactNode;
  className?: string;
}) {
  return (
    <p
      className={cn(
        "grid grid-cols-[auto_minmax(0,1fr)] items-start gap-x-2 font-mono text-[11px] leading-[1.4] tracking-[0.04em]",
        className,
      )}
    >
      <JobMark state={state} className="translate-y-px" />
      <span>
        <span
          className={cn(
            "font-medium tracking-[0.1em] uppercase",
            state === "failed" && "text-thread",
          )}
        >
          {label ?? JOB_LABEL[state]}
        </span>
        {children && <span className="text-fg-2">{"\u00a0· "}{children}</span>}
      </span>
    </p>
  );
}

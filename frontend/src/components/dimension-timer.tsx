import { clock } from "@/lib/time";
import { cn } from "@/lib/utils";

/** The line in miniature for a step label, |— 1:42 —|, with the limit beside it if there is
 * one. The practice page's stopwatch. */
export function DimensionMini({
  seconds,
  limit,
  className,
}: {
  seconds: number;
  limit?: number;
  className?: string;
}) {
  const end = "relative h-2.5 w-5 before:absolute before:inset-x-0 before:top-1/2 before:h-px before:bg-current";
  return (
    <span
      role="timer"
      aria-label={`time taken ${clock(seconds)}`}
      className={cn(
        "inline-flex items-center gap-1.5 font-mono text-xs leading-none font-semibold tracking-normal normal-case tabular-nums",
        className,
      )}
    >
      <span aria-hidden className={cn(end, "border-l border-current")} />
      {clock(seconds)}
      <span aria-hidden className={cn(end, "border-r border-current")} />
      {limit !== undefined && <span className="text-fg-2">{clock(limit)}</span>}
    </span>
  );
}

/** Interview mode's soft limit, drawn as a dimension line from a technical drawing: the bar
 * shortens as time runs down, and past the limit the line turns to the accent and counts on. */
export function DimensionTimer({
  elapsed,
  limit,
  className,
}: {
  elapsed: number;
  limit: number;
  className?: string;
}) {
  const left = limit - elapsed;
  const over = left < 0;

  return (
    <div className={className}>
      <div
        role="timer"
        aria-label={over ? `${clock(-left)} over the limit` : `${clock(left)} left`}
        className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3"
      >
        <div
          className={cn(
            "relative grid h-6 place-items-center before:absolute before:inset-x-0 before:top-1/2 before:h-px after:absolute after:inset-x-0 after:inset-y-[5px] after:border-x after:border-fg",
            over ? "before:h-0.5 before:bg-thread" : "before:bg-fg",
          )}
        >
          <span
            className="absolute inset-x-0 top-[calc(50%-2px)] h-1 origin-left bg-fg"
            style={{ transform: `scaleX(${Math.max(0, left / limit)})` }}
          />
          <span
            className={cn(
              "relative bg-ground px-2.5 font-mono text-[13px] leading-none font-semibold tabular-nums",
              over && "text-thread",
            )}
          >
            {over ? `+${clock(-left)}` : clock(left)}
          </span>
        </div>
        <span className="font-mono text-xs leading-none text-fg-2">{clock(limit)}</span>
      </div>
      <p
        className={cn(
          "mt-2 font-mono text-[11px] leading-snug tracking-[0.06em] uppercase",
          over ? "text-thread" : "text-fg-2",
        )}
      >
        {over ? "The wax is melting · overtime" : `Interview mode · soft limit ${clock(limit)}`}
      </p>
    </div>
  );
}

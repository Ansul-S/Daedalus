import { cn } from "@/lib/utils";

function clock(seconds: number): string {
  const whole = Math.round(Math.abs(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
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

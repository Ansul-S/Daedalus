import { cn } from "@/lib/utils";

// Ariadne's thread: the one line in the accent colour. On the practice page it runs down the
// left, knotted at each step: question, answer, verdict, next.

export function Thread({ className, ...props }: React.ComponentProps<"ol">) {
  return (
    <ol
      className={cn("relative list-none py-7 pr-[clamp(16px,3vw,28px)] pl-[60px]", className)}
      {...props}
    />
  );
}

/** A knot on the thread, with the thread running on to the next knot however tall the step
 * is; the last one, `end`, is left open and the thread stops there. */
export function ThreadStep({
  end = false,
  className,
  ...props
}: React.ComponentProps<"li"> & { end?: boolean }) {
  return (
    <li
      className={cn(
        "relative pb-[34px] last:pb-0 before:absolute before:z-[1] before:-left-[37px] before:size-3.5 before:rounded-full before:shadow-[0_0_0_4px_var(--ground)]",
        end
          ? "before:top-3 before:border-2 before:border-thread before:bg-ground"
          : "before:top-px before:bg-thread after:absolute after:top-2 after:-bottom-5 after:-left-[30px] after:w-0.5 after:bg-thread last:after:hidden",
        className,
      )}
      {...props}
    />
  );
}

/** The step's name, then what it measures; a heading when the step is a part of the page. */
export function StepLabel({
  meta,
  as: Label = "div",
  children,
  className,
}: {
  meta?: React.ReactNode;
  as?: "div" | "h2";
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <Label className={cn("type-label flex flex-wrap items-center gap-x-3.5 gap-y-2", className)}>
      <span>{children}</span>
      {meta && <span className="text-fg-2">{meta}</span>}
    </Label>
  );
}

/** Question → Answer → Verdict → Next, drawn across. */
export function ThreadTimeline({ steps, className }: { steps: string[]; className?: string }) {
  return (
    <ol
      className={cn(
        "relative flex list-none items-center before:absolute before:inset-x-1.5 before:top-1.5 before:h-0.5 before:bg-thread",
        className,
      )}
    >
      {steps.map((step, i) => (
        <li
          key={step}
          className={cn(
            "type-label relative grid flex-1 justify-items-center gap-2.5 text-center text-[10px] tracking-[0.08em] text-fg-2 first:justify-items-start first:text-left last:justify-items-end last:text-right",
            "before:size-3.5 before:rounded-full before:shadow-[0_0_0_3px_var(--ground)]",
            i === steps.length - 1
              ? "before:border-2 before:border-thread before:bg-ground"
              : "before:bg-thread",
          )}
        >
          {step}
        </li>
      ))}
    </ol>
  );
}

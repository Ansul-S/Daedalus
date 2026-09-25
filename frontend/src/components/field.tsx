import { useId } from "react";

import { cn } from "@/lib/utils";

// Form fields on a drawing: a mono label over the control, a note on what it takes, and the
// problem with it, in the accent, when the API refused it.

type Control = {
  id: string;
  "aria-describedby"?: string;
  "aria-invalid"?: true;
};

/** A labelled control. The control is drawn by `children`, which is handed the ids that tie
 * it to its label, hint and error. */
export function Field({
  label,
  hint,
  error,
  className,
  children,
}: {
  label: React.ReactNode;
  hint?: React.ReactNode;
  error?: string | null;
  className?: string;
  children: (control: Control) => React.ReactNode;
}) {
  const id = useId();
  const described = [error && `${id}-error`, hint && `${id}-hint`].filter(Boolean).join(" ");
  return (
    <div className={cn("grid min-w-0 content-start gap-1.5", className)}>
      <label htmlFor={id} className="type-label text-fg-2">
        {label}
      </label>
      {children({
        id,
        "aria-describedby": described || undefined,
        "aria-invalid": error ? true : undefined,
      })}
      {error && (
        <p id={`${id}-error`} className="text-small text-thread">
          {error}
        </p>
      )}
      {hint && (
        <p id={`${id}-hint`} className="text-[0.8125rem] leading-snug text-fg-2">
          {hint}
        </p>
      )}
    </div>
  );
}

/** A native select in a hairline box, with a drawn arrow. */
export function Select({ className, ...props }: React.ComponentProps<"select">) {
  return (
    <span className={cn("relative block min-w-0", className)}>
      <select
        className="h-9 w-full min-w-0 cursor-pointer appearance-none border border-line-2 bg-ground pr-8 pl-2.5 font-mono text-xs tracking-[0.02em] text-fg transition-colors hover:border-fg focus-visible:border-fg aria-invalid:border-thread"
        {...props}
      />
      <span
        aria-hidden
        className="pointer-events-none absolute top-1/2 right-2.5 -translate-y-1/2 font-mono text-[9px] leading-none text-fg-2"
      >
        ▼
      </span>
    </span>
  );
}

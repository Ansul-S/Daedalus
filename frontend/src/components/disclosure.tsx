import { cn } from "@/lib/utils";

/** Something to open when wanted, such as a model answer or a passage: a hairline box with a
 * mono summary and a plus that turns into a cross. */
export function Disclosure({
  summary,
  className,
  children,
}: {
  summary: React.ReactNode;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <details className={cn("group max-w-[48rem] border border-line-2", className)}>
      <summary className="type-label flex cursor-pointer list-none items-center justify-between gap-3 px-4 py-3 hover:bg-surface [&::-webkit-details-marker]:hidden">
        {summary}
        <span aria-hidden className="text-sm leading-none transition-transform group-open:rotate-45">
          +
        </span>
      </summary>
      <div className="border-t border-line-2 px-4 py-3.5">{children}</div>
    </details>
  );
}

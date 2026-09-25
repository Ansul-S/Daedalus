import { cn } from "@/lib/utils";

type Line = { command: string; comment?: string };

/** From nothing to a library to practise: study material, the topic map, questions. */
export const FILL_THE_LABYRINTH: Line[] = [
  { command: 'make ingest SRC="notes.pdf 1706.03762"' },
  { command: "make topics" },
  { command: "make generate N=20" },
];

/** Shell commands on an ink panel, in both themes. */
export function Commands({
  title,
  note,
  lines,
  className,
}: {
  title: string;
  note?: string;
  lines: Line[];
  className?: string;
}) {
  // comments line up in one column after the longest command
  const width = Math.max(0, ...lines.map(({ command }) => command.length));
  return (
    <figure className={cn("m-0 max-w-xl bg-ink text-bone", className)}>
      <figcaption className="flex justify-between gap-3 border-b border-bone/16 px-3.5 py-2 font-mono text-[10px] leading-tight tracking-[0.1em] text-bone/70 uppercase">
        <span>{title}</span>
        {note && <span>{note}</span>}
      </figcaption>
      <pre className="overflow-x-auto p-3.5 font-mono text-[12.5px] leading-[1.8]">
        {lines.map(({ command, comment }) => (
          <div key={command}>
            <span className="text-sinopia-lt select-none">$ </span>
            {command}
            {comment && (
              <span className="text-bone/55">
                {" ".repeat(width - command.length + 2)}# {comment}
              </span>
            )}
          </div>
        ))}
      </pre>
    </figure>
  );
}

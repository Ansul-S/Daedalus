import { useId } from "react";

import { cn } from "@/lib/utils";

// A coin of the treasury: its name round the rim, its Greek letter struck in the middle.
// Minted, it is struck in ochre, the colour of what is earned; until then it is only drawn.

export function Coin({
  name,
  glyph,
  minted,
  className,
}: {
  name: string;
  glyph: string;
  minted: boolean;
  className?: string;
}) {
  const rim = useId();
  return (
    <svg viewBox="-46 -46 92 92" aria-hidden className={cn("block size-[86px]", className)}>
      <defs>
        <path id={rim} d="M -30 0 A 30 30 0 0 1 30 0" />
      </defs>
      <circle
        r={43}
        className={minted ? "fill-gold stroke-ink" : "fill-none stroke-line-2"}
        strokeWidth={minted ? 1.6 : 1.2}
        strokeDasharray={minted ? undefined : "3 3"}
      />
      <circle r={37} className={cn("fill-none", minted ? "stroke-ink" : "stroke-line")} strokeWidth={1} />
      <text
        className={cn(
          "font-mono text-[7.4px] font-medium tracking-[0.14em]",
          minted ? "fill-ink" : "fill-fg-3",
        )}
      >
        <textPath href={`#${rim}`} startOffset="50%" textAnchor="middle">
          {name.toUpperCase()}
        </textPath>
      </text>
      <text
        y={16}
        textAnchor="middle"
        lang="el"
        className={cn(
          "font-serif text-[32px]",
          minted ? "fill-ink font-semibold" : "fill-fg-3 font-normal",
        )}
      >
        {glyph}
      </text>
    </svg>
  );
}

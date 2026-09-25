"use client";

import { toast } from "sonner";

import type { CoinOut, EarnedOut } from "@/client/types.gen";
import { Coin } from "@/components/coin";
import { celebrate } from "@/lib/celebrate";
import { number, xpParts } from "@/lib/progress";

// What a graded answer earned, in ochre, the colour of what is earned: the XP and what made
// it up, a new level, and the coins it minted. Each coin is announced as it is minted, and a
// new level throws up Greek letters.

const STAMP =
  "inline-block bg-gold px-2 pt-[5px] pb-1 font-display leading-none font-bold tracking-[0.05em] text-ink uppercase";

export function Earned({ earned }: { earned: EarnedOut }) {
  const { level } = earned;
  return (
    <div className="mt-4 grid gap-2.5">
      <p className="flex flex-wrap items-center gap-x-3.5 gap-y-2">
        <span className={`${STAMP} text-lg`}>+{earned.xp} XP</span>
        <span className="font-mono text-xs leading-[1.4] tracking-[0.04em] text-fg-2">
          {xpParts(earned.parts, earned.streak).join(" · ")} · {number(earned.total_xp)} XP in all
        </span>
      </p>
      {earned.level_up && (
        <p className="flex flex-wrap items-baseline gap-x-3.5 gap-y-1.5">
          <span className={`${STAMP} text-[0.95rem]`}>Level up</span>
          <span className="font-serif text-[1.45rem] leading-none font-medium italic">{level.name}</span>
          <span className="type-label text-fg-2">
            Level {level.number} of {level.of}
          </span>
        </p>
      )}
      {earned.coins.length > 0 && (
        <ul className="flex flex-wrap items-center gap-x-4 gap-y-2">
          {earned.coins.map((coin) => (
            <li key={coin.id} className="inline-flex items-center gap-2">
              <Coin name={coin.name} glyph={coin.glyph} minted className="size-9" />
              <span className="type-label">Minted · {coin.name}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** A coin just minted, as a toast: struck in ochre on the ink panel. */
function CoinToast({ coin }: { coin: CoinOut }) {
  return (
    <div className="flex w-[356px] max-w-[calc(100vw-32px)] items-center gap-3.5 bg-panel py-3 pr-4 pl-3 text-on-panel">
      <Coin name={coin.name} glyph={coin.glyph} minted className="size-14 shrink-0" />
      <div className="min-w-0">
        <p className="font-mono text-[10px] leading-tight tracking-[0.1em] uppercase opacity-75">
          Coin minted
        </p>
        <p className="mt-1 font-display text-lg leading-none font-bold tracking-[0.05em] uppercase">
          {coin.name}
        </p>
        <p className="mt-1 text-small leading-snug opacity-85">{coin.condition}</p>
      </div>
    </div>
  );
}

/** Announces what an answer earned: a toast for each coin, and the letters for a level. */
export function announce(earned: EarnedOut): void {
  for (const coin of earned.coins) {
    toast.custom(() => <CoinToast coin={coin} />, { duration: 9000 });
  }
  if (earned.level_up) void celebrate();
}

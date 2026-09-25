import type { CoinOut, LevelOut, XpOut } from "@/client/types.gen";
import { dayLabel } from "@/lib/time";

// What practice has earned, put into words. The rules themselves are the API's
// (backend/app/scheduling/progress.py); nothing here adds them up again.

/** 1,284 */
export function number(value: number): string {
  return value.toLocaleString("en");
}

/** 1 day, 5 days */
export function days(count: number): string {
  return `${count} day${count === 1 ? "" : "s"}`;
}

/** How far through its level the XP has come, 0 to 1. The top level is always full. */
export function levelShare(xp: number, level: LevelOut): number {
  if (level.next_start === null) return 1;
  return Math.min(1, Math.max(0, (xp - level.start) / (level.next_start - level.start)));
}

/** 73 to Journeyman, or null at the top level. */
export function toNextLevel(xp: number, level: LevelOut): string | null {
  if (level.next_start === null || level.next_name === null) return null;
  return `${number(level.next_start - xp)} to ${level.next_name}`;
}

/** What an answer's XP was made of, leaving out what it didn't earn. `streak` is the length
 * of the streak on the answer's day. */
export function xpParts(parts: XpOut, streak: number): string[] {
  const found: [number, string][] = [
    [parts.score, "for the score"],
    [parts.answered, "for answering"],
    [parts.due, "for a review that was due"],
    [parts.interview, "inside the time limit"],
    [parts.streak, `for the ${streak}-day thread`],
  ];
  return found.filter(([xp]) => xp > 0).map(([xp, why]) => `${xp} ${why}`);
}

// How far along a coin still to earn is, where the API can count it
const HOW_FAR: Record<string, (have: number, need: number) => string> = {
  theseus: (have, need) => `${days(need - have)} to go`,
  cartographer: (have, need) => `${have} of ${need} sources`,
  knossos: (have, need) => `${have} of ${need} so far`,
  "labyrinth-walker": (have, need) => `${have} of ${need} questions`,
  daedalus: (have, need) => `${number(have)} of ${number(need)} XP`,
};

/** Minted Tue 22 Sep: your first graded answer, or Not yet: a 7-day streak (6 days to go). */
export function coinCaption(coin: CoinOut): string {
  if (coin.minted_on) return `Minted ${dayLabel(coin.minted_on)}: ${coin.condition}`;
  const counted =
    coin.have !== null && coin.need !== null && coin.need > 1
      ? (HOW_FAR[coin.id]?.(coin.have, coin.need) ?? `${coin.have} of ${coin.need}`)
      : null;
  return `Not yet: ${coin.condition}${counted ? ` (${counted})` : ""}`;
}

/** A room's mastery, 0 to 1, as one of the six hatching steps. */
export function masteryStep(mastery: number): number {
  return Math.min(5, Math.max(0, Math.round(mastery * 5)));
}

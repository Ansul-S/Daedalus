import { useId } from "react";

import { cn } from "@/lib/utils";

// Daedalus's wing, the XP bar: fourteen flight feathers along a bone, filling from the
// shoulder to the tip as a level is worked through. The feather being grown is hatched.

type Feather = { x: number; y: number; angle: number; length: number; width: number };

// The shoulder, the wrist and the tip, in a 450 × 200 drawing
const SHOULDER = [34, 70];
const WRIST = [214, 48];
const TIP = [336, 30];

function along(from: number[], to: number[], t: number): [number, number] {
  return [from[0] + (to[0] - from[0]) * t, from[1] + (to[1] - from[1]) * t];
}

function feathers(): { flight: Feather[]; coverts: Feather[] } {
  const flight: Feather[] = [];
  // Eight along the arm, splaying back, then six from the wrist to the tip
  for (let i = 0; i < 8; i++) {
    const t = (i + 0.5) / 8;
    const [x, y] = along(SHOULDER, WRIST, t);
    flight.push({ x, y, angle: 97 - 14 * t, length: 82 + 30 * t, width: 9.5 });
  }
  for (let i = 0; i < 6; i++) {
    const t = (i + 0.5) / 6;
    const [x, y] = along(WRIST, TIP, t);
    flight.push({ x, y, angle: 80 - 54 * t, length: 112 + 10 * Math.sin(t * Math.PI), width: 9 });
  }
  // Short coverts over their roots
  const coverts: Feather[] = [];
  for (let i = 0; i < 11; i++) {
    const t = (i + 0.5) / 11;
    const [x, y] = along(SHOULDER, [TIP[0] - 22, TIP[1] + 6], t);
    coverts.push({ x, y, angle: 100 - 40 * t, length: 34 + 10 * t, width: 7 });
  }
  return { flight, coverts };
}

/** A feather as two curves from its root to its tip, widest two fifths of the way along. */
function featherPath({ x, y, angle, length, width }: Feather): string {
  const dx = Math.cos((angle * Math.PI) / 180);
  const dy = Math.sin((angle * Math.PI) / 180);
  const tip = [x + dx * length, y + dy * length];
  const mid = [x + dx * length * 0.42, y + dy * length * 0.42];
  const side = [-dy * width, dx * width];
  const n = (value: number) => value.toFixed(1);
  return (
    `M${n(x)} ${n(y)} Q${n(mid[0] + side[0])} ${n(mid[1] + side[1])} ${n(tip[0])} ${n(tip[1])}` +
    ` Q${n(mid[0] - side[0])} ${n(mid[1] - side[1])} ${n(x)} ${n(y)}Z`
  );
}

const WING = feathers();
export const FLIGHT_FEATHERS = WING.flight.length;
const FLIGHT = WING.flight.map(featherPath);
const COVERTS = WING.coverts.map(featherPath);
const BONE = `M${SHOULDER.join(" ")} Q${WRIST[0] - 40} ${WRIST[1] - 16} ${WRIST.join(" ")} L${TIP.join(" ")}`;

/** `share` of the level done, 0 to 1: that many feathers are whole, the next one grows. */
export function Wing({ share, className }: { share: number; className?: string }) {
  const hatch = useId();
  const full = Math.floor(Math.min(1, Math.max(0, share)) * FLIGHT_FEATHERS + 1e-9);
  return (
    <svg viewBox="0 0 450 200" aria-hidden className={cn("block h-auto w-full", className)}>
      <defs>
        <pattern
          id={hatch}
          width={4}
          height={4}
          patternUnits="userSpaceOnUse"
          patternTransform="rotate(45)"
        >
          <rect width={4} height={4} className="fill-ground" />
          <line x1={1} y1={0} x2={1} y2={4} className="stroke-fg" strokeWidth={1.1} />
        </pattern>
      </defs>
      <path
        d="M34 70 A 300 300 0 0 1 336 30"
        className="fill-none stroke-line-2"
        strokeWidth={1}
        strokeDasharray="3 4"
      />
      {FLIGHT.map((d, i) =>
        i < full ? (
          <path key={i} d={d} className="fill-fg stroke-ground" strokeWidth={1.3} />
        ) : i === full ? (
          <path key={i} d={d} fill={`url(#${hatch})`} className="stroke-fg" strokeWidth={1.2} />
        ) : (
          <path key={i} d={d} className="fill-ground stroke-fg" strokeWidth={1.1} />
        ),
      )}
      {COVERTS.map((d, i) => (
        <path key={i} d={d} className="fill-ground stroke-fg" strokeWidth={1} />
      ))}
      <path d={BONE} className="fill-none stroke-fg" strokeWidth={4.5} strokeLinecap="round" />
    </svg>
  );
}

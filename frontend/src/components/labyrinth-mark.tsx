import { cn } from "@/lib/utils";

// The Cretan labyrinth, drawn square as on the coins of Knossos: one path through seven
// circuits (1 = outermost), walked 3 · 2 · 1 · 4 · 7 · 6 · 5, then the centre. Circuit k is a
// square of half-size r(k), walked all the way round, the direction alternating; every turn
// to the next circuit happens in the gap at the bottom.
const WALK: [circuit: number, direction: 1 | -1, exit: number | null][] = [
  [3, 1, 2],
  [2, -1, -2],
  [1, 1, 1],
  [4, -1, -1],
  [7, 1, null],
  [6, -1, -2],
  [5, 1, 0.5],
];

function labyrinth(w: number): [number, number][] {
  const r = (k: number) => (8.5 - k) * w;
  const points: [number, number][] = [
    [0, r(1) + 1.1 * w], // the entrance
    [0, r(3)],
  ];
  WALK.forEach(([k, dir, exit], i) => {
    points.push([-dir * r(k), r(k)], [-dir * r(k), -r(k)], [dir * r(k), -r(k)], [dir * r(k), r(k)]);
    const x = exit === null ? dir * r(k) : exit * w;
    if (exit !== null) points.push([x, r(k)]);
    const next = WALK[i + 1];
    points.push([x, next ? r(next[0]) : 0]); // to the next circuit, or into the centre
  });
  points.push([0, 0]);
  return points;
}

const POINTS = labyrinth(10);
export const LABYRINTH_PATH = POINTS.map(([x, y], i) => `${i ? "L" : "M"}${x} ${y}`).join(" ");
// Every segment runs straight up, down or across, so the path's length is exact.
export const LABYRINTH_LENGTH = POINTS.slice(1).reduce(
  (sum, [x, y], i) => sum + Math.abs(x - POINTS[i][0]) + Math.abs(y - POINTS[i][1]),
  0,
);
export const LABYRINTH_VIEWBOX = "-82 -82 164 176";

type Props = {
  className?: string;
  strokeWidth?: number;
  /** Draw Ariadne's thread along the path, in the accent colour, instead of the centre stone. */
  thread?: boolean;
  /** Accessible name; without one the mark is decoration. */
  title?: string;
};

export function LabyrinthMark({ className, strokeWidth = 5, thread = false, title }: Props) {
  return (
    <svg
      viewBox={LABYRINTH_VIEWBOX}
      className={cn("shrink-0", className)}
      role={title ? "img" : undefined}
      aria-label={title}
      aria-hidden={title ? undefined : true}
    >
      <path
        d={LABYRINTH_PATH}
        fill="none"
        stroke="currentColor"
        strokeWidth={strokeWidth}
        strokeLinejoin="miter"
        strokeLinecap="square"
      />
      {thread ? (
        <>
          <path
            data-thread
            d={LABYRINTH_PATH}
            fill="none"
            className="stroke-thread"
            strokeWidth={1.7}
            strokeLinejoin="round"
            strokeLinecap="round"
            strokeDasharray={LABYRINTH_LENGTH}
          />
          <circle cx={0} cy={0} r={3.8} className="fill-thread" />
        </>
      ) : (
        <rect x={-3.5} y={-3.5} width={7} height={7} fill="currentColor" />
      )}
    </svg>
  );
}

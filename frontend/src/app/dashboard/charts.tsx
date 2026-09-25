"use client";

import { useLayoutEffect, useRef, useState } from "react";

import type { DayCountOut, ScoreOut } from "@/client/types.gen";
import { questionNumber } from "@/lib/questions";
import { dateLabel, dayLabel, weekdayOf } from "@/lib/time";

// The dashboard's two charts, ported from the pattern book's sheet 07 and drawn at the width
// they are given, so their lettering stays the size of the page's. Point at a day or an
// answer, or focus the chart and use the arrow keys, to read it; each has a table for screen
// readers too.

const TICK = "font-mono text-[9.5px] font-medium tracking-[0.08em] fill-fg-2";

/** The width of the box the chart is drawn in, followed as it changes. */
function useWidth(initial: number) {
  const box = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(initial);
  useLayoutEffect(() => {
    const element = box.current;
    if (!element) return;
    const measure = () => {
      const found = Math.floor(element.getBoundingClientRect().width);
      if (found > 0) setWidth(found);
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return { box, width };
}

/** The data point being read: the one nearest the pointer, or the one the arrow keys reached. */
function useCursor(count: number, xOf: (index: number) => number) {
  const [at, setAt] = useState<number | null>(null);

  function nearest(x: number): number {
    let best = 0;
    for (let i = 1; i < count; i++) {
      if (Math.abs(xOf(i) - x) < Math.abs(xOf(best) - x)) best = i;
    }
    return best;
  }

  return {
    at,
    target: {
      tabIndex: 0,
      onPointerMove: (event: React.PointerEvent<HTMLElement>) => {
        const left = event.currentTarget.getBoundingClientRect().left;
        if (count > 0) setAt(nearest(event.clientX - left));
      },
      onPointerLeave: () => setAt(null),
      onFocus: () => setAt((current) => current ?? (count > 0 ? count - 1 : null)),
      onBlur: () => setAt(null),
      onKeyDown: (event: React.KeyboardEvent<HTMLElement>) => {
        const moves: Record<string, (index: number) => number> = {
          ArrowLeft: (index) => index - 1,
          ArrowRight: (index) => index + 1,
          Home: () => 0,
          End: () => count - 1,
        };
        const move = moves[event.key];
        if (!move || count === 0) return;
        event.preventDefault();
        setAt((current) => Math.min(count - 1, Math.max(0, move(current ?? count - 1))));
      },
    },
  };
}

/** The value and what it belongs to, over the point being read: an ink panel, value first.
 * Screen readers hear the same through a status line that is always there. */
function Readout({
  shown,
  width,
}: {
  shown: { x: number; y: number; value: string; label: string } | null;
  width: number;
}) {
  return (
    <>
      <p role="status" className="sr-only">
        {shown ? `${shown.value}, ${shown.label}` : ""}
      </p>
      {shown && (
        <div
          aria-hidden
          className="pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-full bg-panel px-2.5 py-1.5 font-mono text-[11px] leading-tight whitespace-nowrap text-on-panel"
          // kept inside the card at either end
          style={{ left: Math.min(Math.max(shown.x, 64), width - 64), top: shown.y - 12 }}
        >
          <span className="font-semibold">{shown.value}</span>
          <span className="ml-2 opacity-75">{shown.label}</span>
        </div>
      )}
    </>
  );
}

/** The last 14 practice days on Ariadne's thread: a knot for a day practised, a tick for a
 * day missed, today larger. Today stays an open knot until it has an answer. */
export function StreakThread({ days, today }: { days: DayCountOut[]; today: string }) {
  const { box, width } = useWidth(560);
  const height = 74;
  const y = 22;
  const first = 24;
  const last = width - 24;
  const step = days.length > 1 ? (last - first) / (days.length - 1) : 0;
  const xOf = (index: number) => first + index * step;
  const { at, target } = useCursor(days.length, xOf);
  const practised = days.filter((day) => day.count > 0).length;
  const shown = at === null ? null : days[at];

  return (
    <div ref={box} className="relative">
      <div
        {...target}
        role="group"
        aria-label={`Practice over the last ${days.length} days: ${practised} practised. Arrow keys read each day.`}
      >
        <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} className="block" aria-hidden>
          <line x1={8} y1={y} x2={width - 8} y2={y} className="stroke-thread" strokeWidth={2} />
          {days.map((day, index) => {
            const x = xOf(index);
            const isToday = day.day === today;
            return (
              <g key={day.day}>
                {day.count > 0 ? (
                  <circle
                    cx={x}
                    cy={y}
                    r={isToday ? 7.5 : 5}
                    className={isToday ? "fill-thread stroke-fg" : "fill-fg"}
                    strokeWidth={isToday ? 1.5 : 0}
                  />
                ) : isToday ? (
                  <circle cx={x} cy={y} r={6} className="fill-ground stroke-thread" strokeWidth={2} />
                ) : (
                  <line x1={x} y1={y - 6} x2={x} y2={y + 6} className="stroke-fg-3" strokeWidth={1.2} />
                )}
                {at === index && (
                  <circle cx={x} cy={y} r={11} className="fill-none stroke-fg" strokeWidth={1} />
                )}
                <text x={x} y={y + 26} textAnchor="middle" className={TICK}>
                  {weekdayOf(day.day).charAt(0)}
                </text>
              </g>
            );
          })}
          {days.length > 0 && (
            <>
              <text x={first} y={y + 46} className={TICK}>
                {dateLabel(days[0].day).toUpperCase()}
              </text>
              <text x={last} y={y + 46} textAnchor="end" className={TICK}>
                TODAY · {dayLabel(today).toUpperCase()}
              </text>
            </>
          )}
        </svg>
      </div>
      <Readout
        width={width}
        shown={
          shown && at !== null
            ? {
                x: xOf(at),
                y: y - 8,
                value:
                  shown.count > 0
                    ? `${shown.count} answer${shown.count === 1 ? "" : "s"}`
                    : shown.day === today
                      ? "not yet"
                      : "no practice",
                label: dayLabel(shown.day),
              }
            : null
        }
      />
      <div className="sr-only">
        <table>
          <caption>Graded answers on each of the last {days.length} days</caption>
          <thead>
            <tr>
              <th scope="col">Day</th>
              <th scope="col">Answers</th>
            </tr>
          </thead>
          <tbody>
            {days.map((day) => (
              <tr key={day.day}>
                <td>{dayLabel(day.day)}</td>
                <td>{day.count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/** The scores of the latest answers, oldest first, on a 0 to 1 scale; the newest in the
 * accent colour with its value beside it. */
export function ScoreTrend({ scores, today }: { scores: ScoreOut[]; today: string }) {
  const { box, width } = useWidth(560);
  const narrow = width < 440;
  const height = 190;
  const left = 36;
  const right = narrow ? 18 : 92;
  // Narrow, the newest score's label gets a band of its own above the plot
  const top = narrow ? 30 : 14;
  const bottom = 28;
  const count = scores.length;
  const xOf = (index: number) =>
    count === 1
      ? left + (width - left - right) / 2
      : left + (index / (count - 1)) * (width - left - right);
  const yOf = (score: number) => top + (1 - score) * (height - top - bottom);
  const { at, target } = useCursor(count, xOf);

  if (count === 0) {
    return <p className="mt-2 text-small text-fg-2">No graded answers yet: the trend starts with the first.</p>;
  }

  const points = scores.map((score, index) => `${xOf(index).toFixed(1)},${yOf(score.score).toFixed(1)}`);
  const newest = scores[count - 1];
  const lx = xOf(count - 1);
  const ly = yOf(newest.score);
  const when = newest.day === today ? "today" : dayLabel(newest.day);
  // Beside the newest point, or over it in the band when there is no room to its right
  const label = narrow
    ? { x: lx + 5, y: 12, anchor: "end" as const }
    : { x: lx + 11, y: ly + 4, anchor: "start" as const };
  const shown = at === null ? null : scores[at];
  const high = scores.reduce((best, score) => Math.max(best, score.score), 0);

  return (
    <div ref={box} className="relative">
      <div
        {...target}
        role="group"
        aria-label={`Scores of your last ${count} answers, oldest first, from ${scores[0].score.toFixed(2)} to ${newest.score.toFixed(2)}; the highest ${high.toFixed(2)}. Arrow keys read each answer.`}
      >
        <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} className="block" aria-hidden>
          {[0, 0.5, 1].map((value) => (
            <g key={value}>
              <line x1={left} x2={width - right} y1={yOf(value)} y2={yOf(value)} className="stroke-line" strokeWidth={1} />
              <text x={left - 8} y={yOf(value) + 3.5} textAnchor="end" className={TICK}>
                {value}
              </text>
            </g>
          ))}
          {at !== null && (
            <line x1={xOf(at)} x2={xOf(at)} y1={top} y2={height - bottom} className="stroke-line-2" strokeWidth={1} />
          )}
          {count > 1 && (
            <>
              <path
                d={`M${xOf(0)},${yOf(0)} L${points.join(" L")} L${lx},${yOf(0)} Z`}
                className="fill-fg opacity-[0.07]"
              />
              <polyline
                points={points.join(" ")}
                className="fill-none stroke-fg"
                strokeWidth={1.75}
                strokeLinejoin="round"
                strokeLinecap="round"
              />
            </>
          )}
          {scores.slice(0, -1).map((score, index) => (
            <circle
              key={score.attempt_id}
              cx={xOf(index)}
              cy={yOf(score.score)}
              r={at === index ? 5 : 3.5}
              className="fill-fg stroke-ground"
              strokeWidth={2}
            />
          ))}
          <circle cx={lx} cy={ly} r={at === count - 1 ? 7 : 5.5} className="fill-thread stroke-ground" strokeWidth={2} />
          <text x={label.x} y={label.y} textAnchor={label.anchor} className="fill-thread font-mono text-[11px] font-semibold">
            {newest.score.toFixed(2)} · {when}
          </text>
          <text x={xOf(0)} y={height - 8} textAnchor="middle" className={TICK}>
            1
          </text>
          {count > 1 && (
            <>
              <text x={lx} y={height - 8} textAnchor="middle" className={TICK}>
                {count}
              </text>
              <text x={(xOf(0) + lx) / 2} y={height - 8} textAnchor="middle" className={TICK}>
                ANSWERS, OLDEST TO NEWEST
              </text>
            </>
          )}
        </svg>
      </div>
      <Readout
        width={width}
        shown={
          shown && at !== null
            ? {
                x: xOf(at),
                y: yOf(shown.score) - 6,
                value: shown.score.toFixed(2),
                label: `${questionNumber(shown.question_id)} · ${dayLabel(shown.day)}`,
              }
            : null
        }
      />
      <div className="sr-only">
        <table>
          <caption>Scores of your last {count} answers, oldest first</caption>
          <thead>
            <tr>
              <th scope="col">Question</th>
              <th scope="col">Day</th>
              <th scope="col">Score</th>
            </tr>
          </thead>
          <tbody>
            {scores.map((score) => (
              <tr key={score.attempt_id}>
                <td>{questionNumber(score.question_id)}</td>
                <td>{dayLabel(score.day)}</td>
                <td>{score.score.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/** What comes due over the next week, in words: the days with nothing due are left out. */
export function ComingDue({ due, today }: { due: DayCountOut[]; today: string }) {
  const coming = due.filter((day) => day.count > 0);
  const when = (day: string) =>
    day === today ? "today" : day === due[1]?.day ? "tomorrow" : `on ${dayLabel(day)}`;
  return (
    <p className="font-mono text-[11px] leading-normal tracking-[0.04em] text-fg-2">
      {coming.length === 0
        ? `No reviews come due in the next ${due.length} days.`
        : `Reviews coming due: ${coming.map((day) => `${day.count} ${when(day.day)}`).join(", ")}.`}
    </p>
  );
}

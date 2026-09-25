"use client";

import { useId, useMemo, useRef, useState } from "react";

import type { MapOut, RoomOut } from "@/client/types.gen";
import { MasteryHatch } from "@/components/hatching";
import { masteryStep } from "@/lib/progress";
import { cn } from "@/lib/utils";

// The labyrinth, as drawn on the pattern book's sheet 07: a room for each topic, hatched as
// it is learnt, walls with doors where the maze joins two rooms, a knot for the reviews due in
// a room, the Minotaur in the weakest, and today's thread from the entrance through the rooms
// answered in. The layout itself comes from the API (backend/app/scheduling/labyrinth.py).

// A room's cell, and the margin round the maze
const CW = 140;
const CH = 120;
const P = 20;
// Mono lettering at 9.5 px: the width of one letter, and the longest line on a name plate
const LETTER = 6.05;
const LINE = 12;

function nameLines(name: string): string[] {
  const lines: string[] = [];
  for (const word of name.toUpperCase().split(" ")) {
    const last = lines.at(-1);
    if (last !== undefined && `${last} ${word}`.length <= LINE) lines[lines.length - 1] = `${last} ${word}`;
    else lines.push(word);
  }
  return lines;
}

function plural(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

/** The room as a screen reader hears it. */
export function roomSummary(room: RoomOut, lair: boolean, visited: boolean): string {
  return [
    room.name,
    plural(room.questions, "question"),
    `${room.practised} practised`,
    `mastery ${room.mastery.toFixed(2)}`,
    room.due > 0 ? `${room.due} due` : null,
    lair ? "the Minotaur's room" : null,
    visited ? "visited today" : null,
  ]
    .filter(Boolean)
    .join(", ");
}

const MOVES = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"]);

// The four hatchings between empty and solid, as SVG patterns
const HATCHES: Record<number, { size: number; cross: boolean }> = {
  1: { size: 10, cross: false },
  2: { size: 5.5, cross: false },
  3: { size: 7, cross: true },
  4: { size: 4, cross: true },
};

export function LabyrinthMap({
  map,
  selected,
  onSelect,
  panelId,
}: {
  map: MapOut;
  selected: number | null;
  onSelect: (cell: number | null) => void;
  /** The panel a room opens, below the map. */
  panelId: string;
}) {
  const id = useId();
  const { columns, rows } = map;
  const byCell = useMemo(() => new Map(map.rooms.map((room) => [room.cell, room])), [map.rooms]);
  const joined = useMemo(() => {
    const open = new Set(map.passages.map(([a, b]) => `${Math.min(a, b)}-${Math.max(a, b)}`));
    return (a: number, b: number) => open.has(`${Math.min(a, b)}-${Math.max(a, b)}`);
  }, [map.passages]);
  const rooms = useRef(new Map<number, SVGGElement>());
  const [focused, setFocused] = useState<number | null>(null);
  // The one room in the tab order: the one last focused, the one open, or the entrance
  const reachable = focused ?? selected ?? map.entrance ?? map.rooms[0]?.cell ?? 0;

  const width = columns * CW + P * 2;
  const height = rows * CH + P * 2 + 30;
  const x = (cell: number) => P + (cell % columns) * CW;
  const y = (cell: number) => P + Math.floor(cell / columns) * CH;
  // Where the thread passes through a room
  const knot = (cell: number): [number, number] => [x(cell) + CW * 0.72, y(cell) + CH * 0.64];
  const visited = new Set(map.visits);

  function move(from: number, key: string): number | null {
    const column = from % columns;
    const steps: Record<string, number | null> = {
      ArrowLeft: column > 0 ? from - 1 : null,
      ArrowRight: column < columns - 1 ? from + 1 : null,
      ArrowUp: from - columns,
      ArrowDown: from + columns,
      Home: map.rooms[0]?.cell ?? null,
      End: map.rooms.at(-1)?.cell ?? null,
    };
    const to = steps[key];
    return to !== null && to !== undefined && byCell.has(to) ? to : null;
  }

  function onKeyDown(event: React.KeyboardEvent<SVGGElement>, cell: number) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onSelect(selected === cell ? null : cell);
      return;
    }
    if (event.key === "Escape" && selected !== null) {
      event.preventDefault();
      onSelect(null);
      return;
    }
    if (!MOVES.has(event.key)) return;
    event.preventDefault();
    const to = move(cell, event.key);
    if (to === null) return;
    setFocused(to);
    rooms.current.get(to)?.focus();
  }

  const walls: [number, number, number, number][] = [];
  // A wall, or a wall with a door in its middle half
  const side = (x1: number, y1: number, x2: number, y2: number, door: boolean) => {
    if (!door) walls.push([x1, y1, x2, y2]);
    else {
      walls.push([x1, y1, x1 + (x2 - x1) * 0.22, y1 + (y2 - y1) * 0.22]);
      walls.push([x1 + (x2 - x1) * 0.78, y1 + (y2 - y1) * 0.78, x2, y2]);
    }
  };
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const cell = row * columns + column;
      const left = P + column * CW;
      const top = P + row * CH;
      if (row === 0) side(left, top, left + CW, top, false);
      if (column === 0) side(left, top, left, top + CH, false);
      side(left + CW, top, left + CW, top + CH, column < columns - 1 && joined(cell, cell + 1));
      side(
        left,
        top + CH,
        left + CW,
        top + CH,
        row < rows - 1 ? joined(cell, cell + columns) : cell === map.entrance,
      );
    }
  }

  const below = P + rows * CH;
  const thread =
    map.entrance !== null && map.thread.length > 0
      ? [[knot(map.entrance)[0], below + 16], ...map.thread.map(knot)]
      : [];
  const lairRoom = map.lair === null ? null : byCell.get(map.lair);
  const route = [...new Set(map.thread)].map((cell) => byCell.get(cell)?.name).filter(Boolean);

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      role="group"
      aria-label={[
        `The labyrinth: ${plural(map.rooms.length, "room")}, one for each topic, hatched by mastery.`,
        lairRoom ? `The Minotaur waits in ${lairRoom.name}.` : "",
        route.length > 0
          ? `Today's thread runs from the entrance through ${route.join(", ")}.`
          : "No thread yet today.",
        "Arrow keys move between rooms; Enter lists a room's questions.",
      ].join(" ")}
      className="block h-auto w-full min-w-[720px]"
    >
      <defs>
        {Object.entries(HATCHES).map(([step, { size, cross }]) => (
          <pattern
            key={step}
            id={`${id}h${step}`}
            width={size}
            height={size}
            patternUnits="userSpaceOnUse"
            patternTransform="rotate(45)"
          >
            <line x1={1} y1={0} x2={1} y2={size} className="stroke-fg" strokeWidth={1.1} />
            {cross && <line x1={0} y1={1} x2={size} y2={1} className="stroke-fg" strokeWidth={1.1} />}
          </pattern>
        ))}
      </defs>

      {/* Cells no topic fills: solid stone, outside the maze */}
      {Array.from({ length: rows * columns }, (_, cell) => cell)
        .filter((cell) => !byCell.has(cell))
        .map((cell) => (
          <rect key={cell} x={x(cell)} y={y(cell)} width={CW} height={CH} className="fill-line" />
        ))}

      {map.rooms.map((room) => {
        const { cell } = room;
        const left = x(cell);
        const top = y(cell);
        const step = masteryStep(room.mastery);
        const lines = nameLines(room.name);
        const plate = Math.max(...lines.map((line) => line.length)) * LETTER + 10;
        const questions = `${room.questions} Q`;
        const lair = cell === map.lair;
        const open = selected === cell;
        return (
          <g
            key={cell}
            ref={(element) => {
              if (element) rooms.current.set(cell, element);
              else rooms.current.delete(cell);
            }}
            id={`${panelId}-room-${cell}`}
            role="button"
            tabIndex={cell === reachable ? 0 : -1}
            aria-label={roomSummary(room, lair, visited.has(cell))}
            aria-expanded={open}
            aria-controls={panelId}
            onClick={() => {
              setFocused(cell);
              onSelect(open ? null : cell);
            }}
            onFocus={() => setFocused(cell)}
            onKeyDown={(event) => onKeyDown(event, cell)}
            className="group/room cursor-pointer outline-none"
          >
            <rect
              x={left + 6}
              y={top + 6}
              width={CW - 12}
              height={CH - 12}
              className={step === 5 ? "fill-fg" : "fill-ground"}
            />
            {step > 0 && step < 5 && (
              <rect
                x={left + 6}
                y={top + 6}
                width={CW - 12}
                height={CH - 12}
                fill={`url(#${id}h${step})`}
              />
            )}
            <rect
              x={left + 12}
              y={top + 12}
              width={plate}
              height={lines.length * 12 + 7}
              className="fill-ground"
            />
            {lines.map((line, i) => (
              <text
                key={i}
                x={left + 17}
                y={top + 24 + i * 12}
                className="fill-fg font-mono text-[9.5px] font-medium tracking-[0.04em]"
              >
                {line}
              </text>
            ))}
            <rect
              x={left + 12}
              y={top + CH - 30}
              width={questions.length * LETTER + 10}
              height={16}
              className="fill-ground"
            />
            <text
              x={left + 17}
              y={top + CH - 18.5}
              className="fill-fg-2 font-mono text-[9.5px] font-medium tracking-[0.04em]"
            >
              {questions}
            </text>
            {room.due > 0 && (
              <>
                <circle
                  cx={left + CW - 22}
                  cy={top + 22}
                  r={9.5}
                  className="origin-center animate-due-pulse fill-none stroke-thread opacity-0 [transform-box:fill-box]"
                  strokeWidth={2}
                />
                <circle cx={left + CW - 22} cy={top + 22} r={9.5} className="fill-thread" />
                <text
                  x={left + CW - 22}
                  y={top + 25.5}
                  textAnchor="middle"
                  className="fill-on-thread font-mono text-[10px] font-bold"
                >
                  {room.due}
                </text>
              </>
            )}
            {lair && (
              <>
                <circle
                  cx={left + 44}
                  cy={Math.max(top + 60, top + 12 + lines.length * 12 + 7 + 17)}
                  r={11}
                  className="fill-fg"
                />
                <text
                  x={left + 44}
                  y={Math.max(top + 60, top + 12 + lines.length * 12 + 7 + 17) + 4.5}
                  textAnchor="middle"
                  lang="el"
                  className="fill-ground font-serif text-[13px] font-semibold"
                >
                  Μ
                </text>
              </>
            )}
            {/* Pointed at, focused or open */}
            <rect
              x={left + 3.5}
              y={top + 3.5}
              width={CW - 7}
              height={CH - 7}
              className={cn(
                "fill-none stroke-transparent group-hover/room:stroke-line-2 group-focus-visible/room:stroke-focus",
                open && "stroke-fg group-hover/room:stroke-fg",
              )}
              strokeWidth={open ? 2 : 2.5}
            />
          </g>
        );
      })}

      <g className="pointer-events-none">
        {walls.map(([x1, y1, x2, y2], i) => (
          <line
            key={i}
            x1={x1}
            y1={y1}
            x2={x2}
            y2={y2}
            className="stroke-fg"
            strokeWidth={3}
            strokeLinecap="square"
          />
        ))}
        {thread.length > 0 && (
          <polyline
            points={thread.map(([px, py]) => `${px.toFixed(1)},${py.toFixed(1)}`).join(" ")}
            className="fill-none stroke-thread"
            strokeWidth={3}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        )}
        {map.visits.map((cell, i) => {
          const [px, py] = knot(cell);
          const last = i === map.visits.length - 1;
          return (
            <circle
              key={`${cell}-${i}`}
              cx={px}
              cy={py}
              r={last ? 6.5 : 4.5}
              className={last ? "fill-thread stroke-fg" : "fill-thread stroke-ground"}
              strokeWidth={last ? 2 : 1.5}
            />
          );
        })}
        {map.entrance !== null && (
          <text
            x={knot(map.entrance)[0]}
            y={below + 28}
            textAnchor="middle"
            className="fill-fg-2 font-mono text-[9px] font-medium tracking-[0.12em]"
          >
            ENTRANCE
          </text>
        )}
      </g>
    </svg>
  );
}

/** The map's key, under it: hatching for mastery in six steps, then the marks. */
export function MapLegend({ className }: { className?: string }) {
  const item = "inline-flex items-center gap-2";
  return (
    <ul
      className={cn(
        "flex flex-wrap gap-x-5 gap-y-2 font-mono text-[10.5px] leading-[1.3] tracking-[0.05em] text-fg-2",
        className,
      )}
    >
      <li className={item}>
        <span aria-hidden className="inline-flex gap-[3px] text-fg">
          {[0, 1, 2, 3, 4, 5].map((step) => (
            <MasteryHatch key={step} level={step} className="size-3 border" />
          ))}
        </span>
        hatching = mastery, 0 to 1
      </li>
      <li className={item}>
        <i aria-hidden className="inline-block size-3 rounded-full bg-thread" />
        reviews due
      </li>
      <li className={item}>
        <i aria-hidden className="inline-block h-[3px] w-[22px] bg-thread" />
        today&apos;s thread
      </li>
      <li className={item}>
        <i
          aria-hidden
          lang="el"
          className="inline-grid size-4 place-items-center rounded-full bg-fg font-serif text-[10px] leading-none font-semibold text-ground not-italic"
        >
          Μ
        </i>
        the Minotaur · the weakest room
      </li>
    </ul>
  );
}

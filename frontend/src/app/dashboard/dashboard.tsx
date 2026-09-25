"use client";

import { useQuery } from "@tanstack/react-query";
import { useId, useState } from "react";

import {
  practiceMapOptions,
  practiceProgressOptions,
  practiceStatsOptions,
} from "@/client/@tanstack/react-query.gen";
import type { MapOut, ProgressOut, StatsOut } from "@/client/types.gen";
import { ApiProblem } from "@/components/api-problem";
import { Coin } from "@/components/coin";
import { Commands, FILL_THE_LABYRINTH } from "@/components/commands";
import { SheetSection, TitleBlock } from "@/components/sheet";
import { Wing } from "@/components/wing";
import { coinCaption, days, levelShare, number, toNextLevel } from "@/lib/progress";
import { cn } from "@/lib/utils";

import { ComingDue, ScoreTrend, StreakThread } from "./charts";
import { LabyrinthMap, MapLegend } from "./labyrinth-map";
import { RoomPanel } from "./room-panel";

// Everything practice has built, as drawn on the pattern book's sheet 07: the labyrinth of
// topics, the wing that grows with each level, the thread of days practised, the scores, and
// the treasury of coins.

const CARD = "min-w-0 border border-line-2 px-[18px] pt-4 pb-[18px]";
const SUB = "type-label text-fg-2";

function Labyrinth({ map, today }: { map: MapOut; today: string }) {
  const [selected, setSelected] = useState<number | null>(null);
  const panel = useId();
  const room = map.rooms.find((found) => found.cell === selected) ?? null;
  const questions = map.rooms.reduce((sum, found) => sum + found.questions, 0);

  function close() {
    const cell = selected;
    setSelected(null);
    // Back to the room the panel belonged to
    if (cell !== null) document.getElementById(`${panel}-room-${cell}`)?.focus();
  }

  return (
    <section aria-labelledby={`${panel}-title`} className={CARD}>
      <h2 id={`${panel}-title`} className={SUB}>
        {map.rooms.length === 0
          ? "The labyrinth · no rooms yet"
          : `The labyrinth · ${map.rooms.length} room${map.rooms.length === 1 ? "" : "s"}, one per topic · ${questions} question${questions === 1 ? "" : "s"}`}
      </h2>
      {map.rooms.length === 0 ? (
        <>
          <p className="mt-3 mb-5 max-w-[62ch]">
            No rooms yet: every topic with questions becomes a room. Add study material, build
            the topic map and write questions from it:
          </p>
          <Commands title="Filling the labyrinth" lines={FILL_THE_LABYRINTH} />
        </>
      ) : (
        <>
          {/* Under 720 px the map keeps its size and scrolls sideways inside the card */}
          <div className="-mx-[18px] mt-1.5 overflow-x-auto px-[18px] pb-1">
            <LabyrinthMap map={map} selected={selected} onSelect={setSelected} panelId={panel} />
          </div>
          <MapLegend className="mt-2.5" />
          <div
            id={panel}
            className="mt-4"
            onKeyDown={(event) => {
              if (event.key !== "Escape" || selected === null) return;
              event.preventDefault();
              close();
            }}
          >
            {room ? (
              <RoomPanel
                key={room.cell}
                room={room}
                lair={room.cell === map.lair}
                today={today}
                onClose={close}
              />
            ) : (
              <p className="font-mono text-[11px] leading-normal tracking-[0.04em] text-fg-2">
                Choose a room to see its questions.
              </p>
            )}
          </div>
        </>
      )}
    </section>
  );
}

function Wings({ progress }: { progress: ProgressOut }) {
  const { xp, level } = progress;
  const share = levelShare(xp, level);
  const next = toNextLevel(xp, level);
  return (
    <section className={CARD}>
      <h2 className={SUB}>The wings · your level</h2>
      <Wing share={share} className="mt-2" />
      <p className="mt-2 flex flex-wrap items-baseline gap-x-3.5 gap-y-1.5">
        <span className="font-serif text-[1.6rem] leading-none font-medium italic">{level.name}</span>
        <span className={SUB}>
          Level {level.number} of {level.of}
        </span>
      </p>
      <p className={cn(SUB, "mt-2")}>
        <b className="font-display text-2xl font-bold tracking-[0.02em] text-fg">{number(xp)}</b> XP
        {next ? ` · ${next} · ${Math.floor(share * 100)}% of this level` : " · the top level"}
      </p>
    </section>
  );
}

function threadTitle({ days: count, today, best }: ProgressOut["streak"]): string {
  if (count > 0) {
    return `The thread holds · ${days(count)}${today ? "" : " · answer today to keep it"}`;
  }
  return best > 0 ? `The thread broke · your best was ${days(best)}` : "The thread · not started yet";
}

function Days({ progress, stats }: { progress: ProgressOut; stats: StatsOut }) {
  const answers = stats.scores.length;
  return (
    <section className={CARD}>
      <h2 className={SUB}>{threadTitle(progress.streak)}</h2>
      <div className="mt-2">
        <StreakThread days={stats.answered} today={stats.today} />
      </div>
      <ComingDue due={stats.due} today={stats.today} />
      <h2 className={cn(SUB, "mt-[18px]")}>
        {answers > 0 ? `Scores · your last ${answers} answer${answers === 1 ? "" : "s"}` : "Scores"}
      </h2>
      <div className="mt-2">
        <ScoreTrend scores={stats.scores} today={stats.today} />
      </div>
    </section>
  );
}

function Treasury({ coins }: { coins: ProgressOut["coins"] }) {
  const minted = coins.filter((coin) => coin.minted_on !== null).length;
  return (
    <section className={CARD}>
      <h2 className={SUB}>
        The treasury · {minted} of {coins.length} coins minted
      </h2>
      <ul className="mt-2.5 grid grid-cols-2 gap-x-3 gap-y-[18px] min-[700px]:grid-cols-5">
        {coins.map((coin) => (
          <li key={coin.id} className="grid content-start justify-items-center gap-1.5 text-center">
            <Coin name={coin.name} glyph={coin.glyph} minted={coin.minted_on !== null} />
            <p className="max-w-[17ch] text-[0.8125rem] leading-[1.35] text-fg-2">
              <b className="block font-display text-[0.95rem] leading-[1.1] font-bold tracking-[0.05em] text-fg uppercase">
                {coin.name}
              </b>
              {coinCaption(coin)}
            </p>
          </li>
        ))}
      </ul>
    </section>
  );
}

export function Dashboard({ children }: { children: React.ReactNode }) {
  const progress = useQuery(practiceProgressOptions());
  const map = useQuery(practiceMapOptions());
  const stats = useQuery(practiceStatsOptions());
  const failed = progress.error ?? map.error ?? stats.error;
  const found = progress.data;

  function retry() {
    for (const query of [progress, map, stats]) if (query.isError) void query.refetch();
  }

  return (
    <>
      <SheetSection aria-busy={progress.isFetching || map.isFetching || stats.isFetching}>
        {children}
        {progress.data && map.data && stats.data ? (
          <div className="grid gap-[18px]">
            <Labyrinth map={map.data} today={stats.data.today} />
            <div className="grid grid-cols-1 gap-[18px] min-[900px]:grid-cols-2">
              <Wings progress={progress.data} />
              <Days progress={progress.data} stats={stats.data} />
            </div>
            <Treasury coins={progress.data.coins} />
          </div>
        ) : failed ? (
          <ApiProblem error={failed} retry={retry} />
        ) : (
          <p className="text-fg-2">Drawing the labyrinth…</p>
        )}
      </SheetSection>
      <TitleBlock
        cells={[
          { label: "Sheet", value: "D-01" },
          { label: "Drawing", value: "Dashboard" },
          { label: "Level", value: found ? `${found.level.number} · ${found.level.name}` : "–" },
          { label: "XP", value: found ? number(found.xp) : "–" },
          { label: "Thread", value: found ? days(found.streak.days) : "–" },
          {
            label: "Coins",
            value: found
              ? `${found.coins.filter((coin) => coin.minted_on !== null).length} of ${found.coins.length}`
              : "–",
          },
        ]}
      />
    </>
  );
}

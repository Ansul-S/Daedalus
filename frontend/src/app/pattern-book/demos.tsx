"use client";

import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";

import { DimensionTimer } from "@/components/dimension-timer";
import { GlyphMosaic } from "@/components/glyph-mosaic";
import { LABYRINTH_LENGTH, LabyrinthMark } from "@/components/labyrinth-mark";
import { Markdown } from "@/components/markdown";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { GRIDS } from "@/lib/glyph/grids";

const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

/** The mark with Ariadne's thread drawn through it, one circuit at a time. */
export function TracedMark() {
  const frame = useRef<HTMLDivElement>(null);

  function trace() {
    const thread = frame.current?.querySelector<SVGPathElement>("[data-thread]");
    if (!thread || reducedMotion()) return;
    thread.style.transition = "none";
    thread.style.strokeDashoffset = String(LABYRINTH_LENGTH);
    thread.getBoundingClientRect(); // commit the start before animating from it
    requestAnimationFrame(() => {
      thread.style.transition = "stroke-dashoffset 3.6s cubic-bezier(.5,.05,.25,1)";
      thread.style.strokeDashoffset = "0";
    });
  }

  return (
    <figure className="m-0 grid justify-items-center gap-4 border border-line-2 p-[clamp(18px,3vw,32px)]">
      <div ref={frame} className="w-[min(300px,100%)]">
        <LabyrinthMark
          thread
          strokeWidth={4.6}
          className="h-auto w-full"
          title="The Daedalus mark: the Cretan labyrinth in square form, with the thread traced through it"
        />
      </div>
      <figcaption className="max-w-[40ch] text-center text-small text-fg-2">
        One path, seven circuits, walked 3 · 2 · 1 · 4 · 7 · 6 · 5, then the centre. The line in
        the accent colour is the thread.
      </figcaption>
      <Button variant="outline" size="sm" onClick={trace}>
        Trace the thread
      </Button>
    </figure>
  );
}

const LIMIT = 180;
const RUN_END = LIMIT + 14;

/** Three minutes run in nine seconds, into overtime. */
export function TimerDemo() {
  const [elapsed, setElapsed] = useState(78);
  const raf = useRef(0);

  useEffect(() => () => cancelAnimationFrame(raf.current), []);

  function run() {
    cancelAnimationFrame(raf.current);
    if (reducedMotion()) {
      setElapsed(RUN_END);
      return;
    }
    const start = performance.now();
    const step = (now: number) => {
      const seconds = Math.min(RUN_END, ((now - start) / 1000) * 20);
      setElapsed(seconds);
      if (seconds < RUN_END) raf.current = requestAnimationFrame(step);
    };
    raf.current = requestAnimationFrame(step);
  }

  return (
    <div className="grid gap-3">
      <DimensionTimer elapsed={elapsed} limit={LIMIT} />
      <Button variant="outline" size="sm" className="justify-self-start" onClick={run}>
        Run 3:00 in 9 seconds
      </Button>
    </div>
  );
}

/** Holroyd's Daedalus at either density, and the reveal on demand. */
export function MosaicLab() {
  const [grid, setGrid] = useState<"daedalus" | "daedalus-coarse">("daedalus");
  const [round, setRound] = useState(0);
  const { cols, rows } = GRIDS[grid];

  return (
    <div className="grid grid-cols-1 gap-[clamp(24px,4vw,48px)] md:grid-cols-2">
      <div className="w-full max-w-[520px]">
        <div className="text-fg outline outline-1 outline-offset-[9px] outline-line-2">
          <GlyphMosaic
            key={`${grid}-${round}`}
            grid={grid}
            reveal={round ? "load" : "view"}
          />
        </div>
        <p className="type-label mt-[18px] text-[10px] tracking-[0.08em] text-fg-2">
          Fig. 3 · Charles Holroyd, <i>Daedalus</i>, 1895 · {cols} × {rows} letters · move over
          it
        </p>
      </div>
      <div className="grid content-start gap-4">
        <fieldset className="border border-line-2 px-3 pt-2.5 pb-3">
          <legend className="type-label px-1.5">Letters across</legend>
          {(
            [
              ["daedalus", "Fine · 168, as on the page"],
              ["daedalus-coarse", "Coarse · 104, for small screens"],
            ] as const
          ).map(([value, label]) => (
            <label key={value} className="mt-2 flex cursor-pointer items-center gap-2 text-small">
              <input
                type="radio"
                name="density"
                value={value}
                checked={grid === value}
                onChange={() => setGrid(value)}
                className="accent-thread"
              />
              {label}
            </label>
          ))}
        </fieldset>
        <Button
          variant="outline"
          size="sm"
          className="justify-self-start"
          onClick={() => setRound((n) => n + 1)}
        >
          Replay denoising
        </Button>
      </div>
    </div>
  );
}

const ANSWER =
  "Attention lets every position look at every other one directly. The scores are dot " +
  "products of queries and keys, scaled by $1/\\sqrt{d_k}$ so the softmax keeps useful " +
  "gradients when $d_k$ is large.";

/** The controls, as the practice page will use them. */
export function ControlsDemo() {
  const [answer, setAnswer] = useState(ANSWER);

  return (
    <div className="grid gap-8">
      <div className="flex flex-wrap items-center gap-3">
        <Button>
          Submit answer <kbd>⌘↵</kbd>
        </Button>
        <Button variant="outline">Next question</Button>
        <Button variant="ghost">Skip</Button>
        <Button variant="destructive">Retire</Button>
        <Button variant="link">See the passage</Button>
        <Button size="sm">Small</Button>
      </div>
      <Tabs defaultValue="write" className="max-w-3xl">
        <div className="type-label flex flex-wrap items-center gap-x-3.5 gap-y-2">
          <span>Your answer</span>
          <TabsList>
            <TabsTrigger value="write">Write</TabsTrigger>
            <TabsTrigger value="preview">Preview</TabsTrigger>
          </TabsList>
        </div>
        <TabsContent value="write">
          <Textarea
            aria-label="Answer"
            value={answer}
            onChange={(event) => setAnswer(event.target.value)}
          />
        </TabsContent>
        <TabsContent value="preview">
          <div className="min-h-32 border border-line-2 bg-surface px-4 py-3.5">
            <Markdown>{answer}</Markdown>
          </div>
        </TabsContent>
      </Tabs>
      <div className="flex flex-wrap items-center gap-6">
        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant="outline" size="sm">
              Point at me
            </Button>
          </TooltipTrigger>
          <TooltipContent>Graded against the two passages above</TooltipContent>
        </Tooltip>
        <Button
          variant="outline"
          size="sm"
          onClick={() => toast("Level up: Craftsman", { description: "460 XP · 140 to Inventor" })}
        >
          Show a toast
        </Button>
        <div className="grid w-56 gap-2">
          <span className="type-label text-fg-2">Progress · 53%</span>
          <Progress value={53} aria-label="Example progress" />
        </div>
      </div>
    </div>
  );
}

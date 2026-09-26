"use client";

import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";

import type { RatingIn, RatingOut } from "@/client/types.gen";
import { DimensionTimer } from "@/components/dimension-timer";
import { Checkbox, Field, Input, Select } from "@/components/field";
import { FileDrop } from "@/components/file-drop";
import { GlyphMosaic } from "@/components/glyph-mosaic";
import { LABYRINTH_LENGTH, LabyrinthMark } from "@/components/labyrinth-mark";
import { Markdown } from "@/components/markdown";
import { Rate } from "@/components/rating";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Wing } from "@/components/wing";
import { GRIDS } from "@/lib/glyph/grids";
import { FILE_TYPES, fileSize } from "@/lib/library";

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

// The third level, Craftsman, runs from 300 XP to Inventor at 600
const CRAFTSMAN = 300;
const INVENTOR = 600;

/** The wing through one level, drawn at whatever XP the slider gives. */
export function WingDemo() {
  const [xp, setXp] = useState(460);
  const share = (xp - CRAFTSMAN) / (INVENTOR - CRAFTSMAN);
  return (
    <div className="grid max-w-[30rem] gap-3">
      <Wing share={share} />
      <p className="font-mono text-[11px] leading-snug tracking-[0.06em] text-fg-2 uppercase">
        {xp} XP · {INVENTOR - xp} to Inventor · {Math.floor(share * 100)}% of this level
      </p>
      <label className="flex items-center gap-3 font-mono text-[10.5px] tracking-[0.06em] text-fg-2 uppercase">
        XP (example)
        <input
          type="range"
          min={CRAFTSMAN}
          max={INVENTOR}
          step={5}
          value={xp}
          onChange={(event) => setXp(Number(event.target.value))}
          className="min-w-0 flex-1 accent-thread"
        />
      </label>
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

// Kept on this page only: nothing is sent to the API
async function keepHere(body: RatingIn): Promise<RatingOut> {
  return {
    id: Date.now(),
    question_id: body.question_id ?? null,
    grade_id: body.grade_id ?? null,
    value: body.value,
    note: body.note ?? null,
    created_at: new Date().toISOString(),
  };
}

const POOR_QUESTION: RatingOut = {
  id: 1,
  question_id: 43,
  grade_id: null,
  value: -1,
  note: "The second key point says the first one again.",
  created_at: "2026-09-25T10:00:00Z",
};

const FAIR_GRADE: RatingOut = {
  id: 2,
  question_id: null,
  grade_id: 3,
  value: 1,
  note: null,
  created_at: "2026-09-25T10:00:00Z",
};

/** A question rated poor with its note, and a fair grade. Press the other words, or press
 * Poor question again to change the note. */
export function RatingDemo() {
  return (
    <div className="flex flex-wrap items-start gap-x-10 gap-y-5">
      <Rate rated="question" id={43} initial={POOR_QUESTION} save={keepHere} />
      <Rate rated="grade" id={3} initial={FAIR_GRADE} save={keepHere} />
    </div>
  );
}

/** A filter, boxes to type in and to tick, and a field the API refused, as the question bank
 * and the library draw them. */
export function FieldsDemo() {
  const [topic, setTopic] = useState("");
  const [count, setCount] = useState("10");
  const [formulas, setFormulas] = useState(true);
  return (
    <div className="grid max-w-3xl grid-cols-1 items-start gap-x-6 gap-y-4 sm:grid-cols-[14rem_minmax(0,1fr)]">
      <Field label="Topic">
        {(control) => (
          <Select {...control} value={topic} onChange={(event) => setTopic(event.target.value)}>
            <option value="">All topics</option>
            <option value="attention">attention (6)</option>
            <option value="vanishing gradient">vanishing gradient (2)</option>
          </Select>
        )}
      </Field>
      <Field
        label="arXiv ID or address"
        hint="1706.03762, or 1706.03762v7 for one version; its arxiv.org address works too"
      >
        {(control) => <Input {...control} placeholder="1706.03762" spellCheck={false} />}
      </Field>
      <Field label="How many">
        {(control) => (
          <Input
            {...control}
            type="number"
            min={1}
            max={100}
            value={count}
            onChange={(event) => setCount(event.target.value)}
          />
        )}
      </Field>
      <Checkbox
        label="Formulas as LaTeX"
        hint="PDFs only; slower"
        checked={formulas}
        onChange={(event) => setFormulas(event.target.checked)}
        className="sm:pt-6"
      />
      <Field
        label="Its quote, word for word"
        hint="Six or more words copied from the passage, showing the point is there."
        error="the quote is not in chunk 153 (closest match 94%)"
      >
        {(control) => (
          <Textarea
            {...control}
            defaultValue="The retriever finds relevant chunks, and the reader pulls the answer out of them."
            className="min-h-14 py-2.5 text-small"
          />
        )}
      </Field>
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

/** Files dropped or chosen, listed as the library lists them before they are sent. Nothing
 * leaves the page. */
export function FileDropDemo() {
  const [files, setFiles] = useState<File[]>([]);
  return (
    <div className="grid max-w-md grid-cols-1 gap-3">
      <FileDrop accept={FILE_TYPES} onFiles={(chosen) => setFiles((kept) => [...kept, ...chosen])}>
        Drop PDFs or notebooks here, or
      </FileDrop>
      {files.length > 0 && (
        <ul aria-label="Chosen" className="border-t border-line-2">
          {files.map((file, i) => (
            <li
              key={i}
              className="flex justify-between gap-3 border-b border-dotted border-line-2 py-1 font-mono text-[11px] leading-[1.4]"
            >
              <span className="min-w-0 truncate">{file.name}</span>
              <span className="shrink-0 text-fg-2">{fileSize(file.size)}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

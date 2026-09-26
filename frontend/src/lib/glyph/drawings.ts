import { LABYRINTH_PATH } from "@/components/labyrinth-mark";

import { RAMP } from "./grids";

// Line drawings made in code, redrawn in Greek letters as the etchings are. Each draws itself in
// grey on white; the mosaic averages it cell by cell. A drawing is not a picture of paper, so it
// never flips on a dark ground: its lines are simply drawn in the light ink there.

type Draw = (x: CanvasRenderingContext2D, width: number, height: number) => void;

/** A few numbers that are the same on every visit: a page's line lengths. */
function seeded(seed: number): () => number {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** One turn of the Greek key, standing on the lower rule at `base`, in units of `u`. */
function keyTurn(x: CanvasRenderingContext2D, left: number, base: number, u: number) {
  x.moveTo(left + 4 * u, base);
  x.lineTo(left + 4 * u, base - 10 * u);
  x.lineTo(left + 18 * u, base - 10 * u);
  x.lineTo(left + 18 * u, base - 4 * u);
  x.lineTo(left + 9 * u, base - 4 * u);
  x.lineTo(left + 9 * u, base - 7 * u);
  x.lineTo(left + 14 * u, base - 7 * u);
}

export const DRAWINGS = {
  /** A page of a paper: its title, then lines of text in two paragraphs. */
  page(x, width, height) {
    x.fillStyle = "#f2f2f2";
    x.fillRect(0, 0, width, height);
    const margin = width * 0.1;
    const line = height / 12;
    const random = seeded(7);
    let y = height * 0.12;
    x.fillStyle = "#111";
    x.fillRect(margin, y, width * 0.52, line * 0.95);
    y += line * 1.9;
    for (let i = 0; i < 8; i++) {
      const last = i % 4 === 3; // a paragraph's last line is short
      const length = (last ? 0.42 : 0.74 + random() * 0.1) * (width - 2 * margin);
      x.fillStyle = last ? "#5a5a5a" : "#262626";
      x.fillRect(margin, y, length, line * 0.42);
      y += line * 0.95;
    }
  },
  /** The meander, the Greek key, between two rules. */
  meander(x, width, height) {
    x.fillStyle = "#f2f2f2";
    x.fillRect(0, 0, width, height);
    const u = height / 17;
    const base = height * 0.5 + 6 * u;
    x.strokeStyle = "#111";
    x.lineWidth = u * 1.25;
    x.lineCap = "square";
    x.lineJoin = "miter";
    x.beginPath();
    x.moveTo(0, base);
    x.lineTo(width, base);
    x.moveTo(0, base - 13 * u);
    x.lineTo(width, base - 13 * u);
    for (let left = -24 * u; left < width; left += 24 * u) keyTurn(x, left, base, u);
    x.stroke();
  },
  /** The Cretan labyrinth, the mark in the header. */
  labyrinth(x, width, height) {
    x.fillStyle = "#f4f4f4";
    x.fillRect(0, 0, width, height);
    const scale = Math.min(width / 176, height / 186) * 0.92;
    x.save();
    x.translate(width / 2, height / 2 - 5 * scale);
    x.scale(scale, scale);
    x.lineJoin = "miter";
    x.lineCap = "square";
    x.strokeStyle = "#101010";
    x.lineWidth = 6;
    x.stroke(new Path2D(LABYRINTH_PATH));
    x.fillStyle = "#000";
    x.fillRect(-5, -5, 10, 10);
    x.restore();
  },
} satisfies Record<string, Draw>;

export type DrawingName = keyof typeof DRAWINGS;

const TOP = RAMP.length - 1;

/** The step whose ink matches a darkness from 0 to 1; a little grey still counts as paper. */
function stepFor(darkness: number): number {
  const target = Math.min(1, Math.max(0, (darkness - 0.06) / 0.91)) * RAMP[TOP].coverage;
  let best = 0;
  for (let step = 1; step <= TOP; step++) {
    if (Math.abs(RAMP[step].coverage - target) < Math.abs(RAMP[best].coverage - target)) {
      best = step;
    }
  }
  return best;
}

// Each cell is sampled over this many pixels across and down: a cell is 0.6 as wide as tall.
const SAMPLE_X = 3;
const SAMPLE_Y = 5;

/** A drawing's steps on a grid of `cols` by `rows` cells, row by row. */
export function drawSteps(name: DrawingName, cols: number, rows: number): Uint8Array {
  const width = cols * SAMPLE_X;
  const height = rows * SAMPLE_Y;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const x = canvas.getContext("2d", { willReadFrequently: true })!;
  DRAWINGS[name](x, width, height);
  const pixels = x.getImageData(0, 0, width, height).data;
  const steps = new Uint8Array(cols * rows);
  for (let cy = 0; cy < rows; cy++) {
    for (let cx = 0; cx < cols; cx++) {
      let light = 0;
      for (let y = cy * SAMPLE_Y; y < (cy + 1) * SAMPLE_Y; y++) {
        let at = (y * width + cx * SAMPLE_X) * 4;
        for (let i = 0; i < SAMPLE_X; i++, at += 4) {
          light += 0.2126 * pixels[at] + 0.7152 * pixels[at + 1] + 0.0722 * pixels[at + 2];
        }
      }
      steps[cy * cols + cx] = stepFor(1 - light / (SAMPLE_X * SAMPLE_Y * 255));
    }
  }
  return steps;
}

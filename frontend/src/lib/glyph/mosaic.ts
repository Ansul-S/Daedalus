import { GRIDS, RAMP, type GridName } from "./grids";

// Draws a glyph grid on a canvas, one letter per cell, and reveals it the way a diffusion model
// denoises: every cell starts as a random step and settles on its own, the top rows first.

const TOP = RAMP.length - 1;
const REVEAL_MS = 950;
const REDUCED = "(prefers-reduced-motion: reduce)";

/** A grid's steps, 0 (paper) to 12 (ink), row by row. */
export function decodeGrid(name: GridName): Uint8Array {
  const { cols, rows, data } = GRIDS[name];
  const steps = new Uint8Array(cols * rows);
  let at = 0;
  let i = 0;
  while (i < data.length && at < steps.length) {
    const step = data.charCodeAt(i++) - 97;
    let run = 0;
    while (i < data.length) {
      const c = data.charCodeAt(i);
      if (c < 48 || c > 57) break;
      run = run * 10 + (c - 48);
      i++;
    }
    run ||= 1;
    steps.fill(step, at, Math.min(steps.length, at + run));
    at += run;
  }
  return steps;
}

export class Mosaic {
  private readonly ctx: CanvasRenderingContext2D;
  private readonly base: Uint8Array;
  private readonly cols: number;
  private readonly rows: number;
  private target = new Uint8Array(0);
  private cur = new Uint8Array(0);
  private lockAt = new Float64Array(0);
  private locked = new Uint8Array(0);
  private atlas: HTMLCanvasElement | null = null;
  private boxW = 0;
  private boxH = 0;
  private cellW = 0;
  private cellH = 0;
  private glyphW = 0;
  private glyphH = 0;
  private dpr = 1;
  private frame = 0;
  private raf = 0;
  private running = false;
  private disposed = false;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly box: HTMLElement,
    grid: GridName,
    private readonly onProgress?: (done: number) => void,
  ) {
    this.ctx = canvas.getContext("2d")!;
    this.base = decodeGrid(grid);
    this.cols = GRIDS[grid].cols;
    this.rows = GRIDS[grid].rows;
  }

  /** Waits for the mono font's Greek letters: a canvas can't fall back once it has drawn. */
  async fontsReady(): Promise<void> {
    const family = getComputedStyle(this.canvas).fontFamily;
    try {
      await Promise.all([
        document.fonts.load(`400 16px ${family}`, "·:τεαθ"),
        document.fonts.load(`700 16px ${family}`, "ΔΨΘσ"),
      ]);
    } catch {
      // draw with whatever font is there
    }
  }

  get reducedMotion(): boolean {
    return matchMedia(REDUCED).matches;
  }

  /** Whether the box has changed size since the picture was laid out. A ResizeObserver also
   * reports once when it starts watching, which must not cut short a reveal. */
  resized(): boolean {
    const rect = this.box.getBoundingClientRect();
    return Math.abs(rect.width - this.boxW) >= 1 || Math.abs(rect.height - this.boxH) >= 1;
  }

  /** Sizes the canvas to its box and draws the picture, settled or revealed. */
  layout(reveal: boolean): void {
    if (this.disposed) return;
    const rect = this.box.getBoundingClientRect();
    this.boxW = rect.width;
    this.boxH = rect.height;
    const width = Math.max(40, rect.width);
    const height = Math.max(40, rect.height);
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.cellW = width / this.cols;
    this.cellH = height / this.rows;
    this.canvas.width = Math.round(width * this.dpr);
    this.canvas.height = Math.round(height * this.dpr);
    const n = this.cols * this.rows;
    this.cur = new Uint8Array(n);
    this.lockAt = new Float64Array(n);
    this.locked = new Uint8Array(n);
    this.computeTargets();
    this.buildAtlas();
    if (reveal && !this.reducedMotion) this.play();
    else this.settle();
  }

  /** The theme changed: new ink, and on a dark ground the steps flip so the picture stays
   * positive. */
  recolor(): void {
    if (!this.atlas || this.disposed) return;
    this.computeTargets();
    this.buildAtlas();
    if (this.running) this.drawAll();
    else this.settle();
  }

  settle(): void {
    this.cur.set(this.target);
    this.locked.fill(1);
    this.drawAll();
    this.onProgress?.(1);
  }

  /** The denoising reveal. */
  play(): void {
    if (this.disposed || !this.atlas) return;
    const now = performance.now();
    for (let i = 0; i < this.cur.length; i++) {
      const row = Math.floor(i / this.cols) / this.rows;
      this.lockAt[i] = now + REVEAL_MS * (0.1 + 0.62 * Math.random() + 0.28 * row);
      this.locked[i] = 0;
      this.cur[i] = Math.floor(Math.random() * RAMP.length);
    }
    this.drawAll();
    this.start();
  }

  /** The pointer stirs the letters under it, and they settle again. */
  stir(clientX: number, clientY: number): void {
    if (this.reducedMotion || !this.atlas) return;
    const rect = this.canvas.getBoundingClientRect();
    const cx = Math.floor((clientX - rect.left) / this.cellW);
    const cy = Math.floor((clientY - rect.top) / this.cellH);
    const rx = Math.max(4, Math.round(this.cols / 24));
    const ry = Math.max(2, Math.round((rx * this.cellW) / this.cellH));
    const now = performance.now();
    for (let y = Math.max(0, cy - ry); y <= Math.min(this.rows - 1, cy + ry); y++) {
      for (let x = Math.max(0, cx - rx); x <= Math.min(this.cols - 1, cx + rx); x++) {
        const dx = (x - cx) / rx;
        const dy = (y - cy) / ry;
        if (dx * dx + dy * dy > 1) continue;
        const i = y * this.cols + x;
        if (this.locked[i]) {
          this.locked[i] = 0;
          this.lockAt[i] = now + 140 + Math.random() * 420;
        }
      }
    }
    this.start();
  }

  dispose(): void {
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    this.running = false;
  }

  private computeTargets(): void {
    const flip = getComputedStyle(this.canvas).getPropertyValue("--art-invert").trim() === "1";
    this.target = Uint8Array.from(this.base, (step) => (flip ? TOP - step : step));
  }

  // One cell-sized drawing of each step, copied into place: far faster than fillText per cell.
  private buildAtlas(): void {
    const { dpr, cellW, cellH } = this;
    const glyphW = Math.max(1, Math.ceil(cellW * dpr));
    const glyphH = Math.max(1, Math.ceil(cellH * dpr));
    const atlas = document.createElement("canvas");
    atlas.width = glyphW * RAMP.length;
    atlas.height = glyphH;
    const x = atlas.getContext("2d")!;
    const style = getComputedStyle(this.canvas);
    const size = Math.min(cellW / 0.6, cellH) * dpr;
    x.textAlign = "center";
    x.textBaseline = "middle";
    RAMP.forEach(({ glyph, weight, reversed }, i) => {
      if (glyph === " ") return;
      x.font = `${weight} ${size.toFixed(2)}px ${style.fontFamily}`;
      const cx = i * glyphW + glyphW / 2;
      const cy = glyphH / 2 + dpr * 0.25;
      x.fillStyle = style.color;
      if (reversed) {
        x.fillRect(i * glyphW, 0, glyphW, glyphH);
        x.globalCompositeOperation = "destination-out";
        x.fillText(glyph, cx, cy);
        x.globalCompositeOperation = "source-over";
      } else {
        x.fillText(glyph, cx, cy);
      }
    });
    this.atlas = atlas;
    this.glyphW = glyphW;
    this.glyphH = glyphH;
  }

  private paint(i: number, clear: boolean): void {
    const { cols, cellW, cellH, dpr } = this;
    const cx = i % cols;
    const cy = (i - cx) / cols;
    // integer cell bounds, so neighbouring cells never leave a seam
    const x0 = Math.round(cx * cellW * dpr);
    const x1 = Math.round((cx + 1) * cellW * dpr);
    const y0 = Math.round(cy * cellH * dpr);
    const y1 = Math.round((cy + 1) * cellH * dpr);
    if (clear) this.ctx.clearRect(x0, y0, x1 - x0, y1 - y0);
    const step = this.cur[i];
    if (step && this.atlas) {
      this.ctx.drawImage(
        this.atlas,
        step * this.glyphW,
        0,
        this.glyphW,
        this.glyphH,
        x0,
        y0,
        x1 - x0,
        y1 - y0,
      );
    }
  }

  private drawAll(): void {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    for (let i = 0; i < this.cur.length; i++) if (this.cur[i]) this.paint(i, false);
  }

  private start(): void {
    if (this.running || this.disposed) return;
    this.running = true;
    this.raf = requestAnimationFrame(this.tick);
  }

  private tick = (now: number): void => {
    let pending = 0;
    this.frame++;
    for (let i = 0; i < this.cur.length; i++) {
      if (this.locked[i]) continue;
      const left = this.lockAt[i] - now;
      if (left <= 0) {
        this.locked[i] = 1;
        if (this.cur[i] !== this.target[i]) {
          this.cur[i] = this.target[i];
          this.paint(i, true);
        }
        continue;
      }
      pending++;
      if ((this.frame + i) & 1) continue; // each cell changes every other frame
      // the noise narrows around the cell's own step as its time comes
      const spread = Math.max(1, Math.round(Math.min(1, left / REVEAL_MS) * RAMP.length));
      const step = Math.min(
        TOP,
        Math.max(0, this.target[i] + Math.round((Math.random() * 2 - 1) * spread)),
      );
      if (step !== this.cur[i]) {
        this.cur[i] = step;
        this.paint(i, true);
      }
    }
    this.onProgress?.(1 - pending / this.cur.length);
    if (pending && !this.disposed) this.raf = requestAnimationFrame(this.tick);
    else this.running = false;
  };
}

"use client";

import { useEffect, useRef } from "react";

import { GRIDS, type GridName } from "@/lib/glyph/grids";
import { Mosaic } from "@/lib/glyph/mosaic";
import { watchTheme } from "@/lib/theme";
import { cn } from "@/lib/utils";

type Props = {
  grid: GridName;
  /** When the denoising reveal plays: on first sight, at once, or never. Never under reduced
   * motion, where the picture simply appears. */
  reveal?: "view" | "load" | "none";
  /** A credit line under the picture; the reveal's progress shows beside it. */
  caption?: React.ReactNode;
  className?: string;
};

const STEPS = 50;

function progress(done: number): string {
  return done >= 1
    ? `denoised · ${STEPS} / ${STEPS} steps`
    : `denoising · step ${Math.max(1, Math.round(done * STEPS))} / ${STEPS}`;
}

/** A picture redrawn in Greek letters, in the colour of the text around it. On a dark ground
 * the steps flip (the --art-invert token), so the picture stays positive. */
export function GlyphMosaic({ grid, reveal = "view", caption, className }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const status = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const mosaic = new Mosaic(canvas.current!, box.current!, grid, (done) => {
      if (status.current) status.current.textContent = progress(done);
    });
    let resize = 0;
    const onResize = new ResizeObserver(() => {
      if (!mosaic.resized()) return;
      clearTimeout(resize);
      resize = window.setTimeout(() => mosaic.layout(false), 100);
    });
    const onView = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) return;
        onView.disconnect();
        if (!mosaic.reducedMotion) mosaic.play();
      },
      { threshold: 0.25 },
    );
    const stir = (event: PointerEvent) => mosaic.stir(event.clientX, event.clientY);
    const stopWatching = watchTheme(() => requestAnimationFrame(() => mosaic.recolor()));
    const element = canvas.current!;
    const frame = box.current!;
    let mounted = true;

    mosaic.fontsReady().then(() => {
      if (!mounted) return;
      mosaic.layout(reveal === "load");
      onResize.observe(frame);
      if (reveal === "view") onView.observe(element);
      element.addEventListener("pointermove", stir);
    });

    return () => {
      mounted = false;
      mosaic.dispose();
      clearTimeout(resize);
      onResize.disconnect();
      onView.disconnect();
      stopWatching();
      element.removeEventListener("pointermove", stir);
    };
  }, [grid, reveal]);

  const { cols, rows } = GRIDS[grid];
  return (
    <figure className={cn("m-0 w-full", className)}>
      <div
        ref={box}
        className="relative w-full"
        style={{ aspectRatio: `${(cols * 0.6).toFixed(1)} / ${rows}` }}
      >
        <canvas
          ref={canvas}
          aria-hidden="true"
          className="absolute inset-0 block size-full touch-pan-y font-mono"
        />
      </div>
      {caption && (
        <figcaption className="type-label mt-2.5 flex flex-wrap justify-between gap-x-4 gap-y-1.5 text-[10px] tracking-[0.08em]">
          <span>{caption}</span>
          <span ref={status}>{progress(1)}</span>
        </figcaption>
      )}
    </figure>
  );
}

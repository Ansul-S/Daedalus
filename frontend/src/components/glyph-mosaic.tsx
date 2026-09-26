"use client";

import { useEffect, useRef } from "react";

import type { DrawingName } from "@/lib/glyph/drawings";
import { GRIDS, type GridName } from "@/lib/glyph/grids";
import { Mosaic, type Picture } from "@/lib/glyph/mosaic";
import { watchTheme } from "@/lib/theme";
import { cn } from "@/lib/utils";

type Props = {
  /** When the denoising reveal plays: on first sight, at once, or never. Never under reduced
   * motion, where the picture simply appears. */
  reveal?: "view" | "load" | "none";
  /** A credit line under the picture; the reveal's progress shows beside it. */
  caption?: React.ReactNode;
  className?: string;
  /** For the picture's box alone, without its caption: a frame, say. */
  boxClassName?: string;
} & (
  | {
      /** An etching's grid, and a coarser one of the same picture for narrow boxes. */
      grid: GridName;
      coarse?: GridName;
      drawing?: never;
    }
  | {
      /** A drawing made in code, `cols` letters across a box of the given `aspect` ("4 / 3"). */
      drawing: DrawingName;
      cols: number;
      aspect: string;
      grid?: never;
    }
);

const STEPS = 50;

function progress(done: number): string {
  return done >= 1
    ? `denoised · ${STEPS} / ${STEPS} steps`
    : `denoising · step ${Math.max(1, Math.round(done * STEPS))} / ${STEPS}`;
}

/** A picture redrawn in Greek letters, in the colour of the text around it. On a dark ground
 * an etching's steps flip (the --art-invert token), so the picture stays positive. */
export function GlyphMosaic({
  reveal = "view",
  caption,
  className,
  boxClassName,
  ...props
}: Props) {
  const box = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const status = useRef<HTMLSpanElement>(null);
  // taken apart, so the picture is drawn again only when one of them changes
  const { grid, drawing } = props;
  const coarse = props.drawing === undefined ? props.coarse : undefined;
  const cols = props.drawing === undefined ? undefined : props.cols;

  useEffect(() => {
    const picture: Picture =
      drawing === undefined ? { grid: grid!, coarse } : { drawing, cols: cols! };
    const mosaic = new Mosaic(canvas.current!, box.current!, picture, (done) => {
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
  }, [grid, coarse, drawing, cols, reveal]);

  const aspect =
    props.drawing === undefined
      ? `${(GRIDS[props.grid].cols * 0.6).toFixed(1)} / ${GRIDS[props.grid].rows}`
      : props.aspect;
  return (
    <figure className={cn("m-0 w-full", className)}>
      <div
        ref={box}
        className={cn("relative w-full", boxClassName)}
        style={{ aspectRatio: aspect }}
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

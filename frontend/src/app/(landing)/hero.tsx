import Link from "next/link";

import { Commands, QUICK_START } from "@/components/commands";
import { GlyphMosaic } from "@/components/glyph-mosaic";
import { Button } from "@/components/ui/button";

import { EnterShortcut } from "./enter-shortcut";

export const HEADLINE = "Practice that remembers what you forget";

/** The front of the sheet, on the field: what Daedalus is, the way in, and Holroyd's Daedalus
 * in Greek letters, drawn in the field's own type colour. */
export function Hero() {
  return (
    <section
      aria-labelledby="hero-title"
      className="bg-field text-on-field [--focus:var(--on-field)] selection:bg-on-field selection:text-field"
    >
      <div className="grid grid-cols-1 items-center gap-[clamp(26px,4vw,56px)] px-[clamp(18px,4vw,48px)] pt-[clamp(28px,5vw,64px)] pb-[clamp(26px,4vw,48px)] min-[920px]:grid-cols-2">
        <div className="min-w-0">
          <p className="type-label mb-[18px] leading-[1.45]">
            Sheet 00 · AI/ML interview practice
          </p>
          <h1 id="hero-title" className="type-hero">
            {HEADLINE}
          </h1>
          <p className="mt-[22px] max-w-[34rem] text-[clamp(1.0625rem,1.4vw,1.25rem)] leading-normal">
            Questions written from your own PDFs, notebooks and arXiv papers, asked again just
            before you would forget them. Every answer is checked, claim by claim, against the
            passages the question came from.
          </p>
          <div className="mt-[26px] flex flex-wrap gap-2.5">
            <Button asChild variant="outline" className="border-on-field bg-on-field text-field">
              <Link href="/practice" aria-keyshortcuts="Enter">
                Enter the labyrinth <kbd aria-hidden>↵</kbd>
              </Link>
            </Button>
            <Button asChild variant="outline">
              <a href="#how-it-works">How it works</a>
            </Button>
          </div>
          <EnterShortcut href="/practice" />
          <Commands
            title="Quick start · local"
            note="free tiers only"
            lines={QUICK_START}
            className="mt-[30px]"
          />
        </div>
        <GlyphMosaic
          grid="daedalus"
          coarse="daedalus-coarse"
          reveal="load"
          className="max-w-[540px] min-[920px]:justify-self-end [&_figcaption]:mt-[18px]"
          boxClassName="text-on-field outline outline-1 outline-offset-[9px] outline-on-field/45 [--art-invert:1]"
          caption={
            <>
              Fig. 1 · Charles Holroyd, <i>Daedalus</i>, etching, 1895 · in Greek letters · move
              over it
            </>
          }
        />
      </div>
    </section>
  );
}

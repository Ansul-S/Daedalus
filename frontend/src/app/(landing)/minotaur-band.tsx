import Link from "next/link";

import { GlyphMosaic } from "@/components/glyph-mosaic";
import { Button } from "@/components/ui/button";

// The dashboard in words, beside Tempesta's Theseus and the Minotaur.
const ROWS = [
  { label: "Rooms", value: "one for each topic in your library" },
  { label: "The lair", value: "the room with the lowest mastery", lair: true },
  { label: "Minotaur slayer", value: "a coin for 0.9 on a question you once scored under 0.4" },
];

export function MinotaurBand() {
  return (
    <section
      aria-labelledby="minotaur-title"
      className="grid grid-cols-1 items-center gap-x-[clamp(24px,4vw,56px)] gap-y-8 bg-panel p-[clamp(24px,4vw,48px)] text-on-panel [--focus:var(--sinopia-lt)] md:grid-cols-2"
    >
      {/* A print mounted on the panel: ink on bone in both themes. Flipped onto the dark panel,
          the etching's light line-work would sink into its hatched arena. */}
      <GlyphMosaic
        grid="minotaur"
        coarse="minotaur-coarse"
        className="max-w-[460px] text-on-panel/66 [&_figcaption]:mt-[18px]"
        boxClassName="bg-bone text-ink outline outline-1 outline-offset-[9px] outline-on-panel/45 [--art-invert:0]"
        caption={
          <>
            Fig. 2 · Antonio Tempesta, <i>Theseus and the Minotaur</i>, etching, after 1606 · in
            Greek letters
          </>
        }
      />
      <div className="grid max-w-[34rem] content-start gap-4">
        <p className="type-label text-sinopia-lt">In the app · the dashboard</p>
        <h2
          id="minotaur-title"
          className="font-display text-[clamp(2.2rem,4.4vw,3.6rem)] leading-[0.88] font-extrabold tracking-[0.005em] uppercase"
        >
          The Minotaur waits in your weakest room
        </h2>
        <p className="text-on-panel/82">
          Every topic is a room of the labyrinth, and practice fills in the rooms you know. The
          room you know least is the Minotaur&apos;s lair. Clear the reviews that are due, then go
          in and face it.
        </p>
        <ul className="border-t border-on-panel/22">
          {ROWS.map(({ label, value, lair }) => (
            <li
              key={label}
              className="grid grid-cols-1 gap-x-[18px] gap-y-1 border-b border-on-panel/22 py-[9px] text-small min-[420px]:grid-cols-[auto_minmax(0,1fr)] min-[420px]:items-baseline"
            >
              <span className="inline-flex items-center gap-2 font-mono text-[10.5px] leading-[1.4] tracking-[0.1em] whitespace-nowrap text-on-panel/62 uppercase">
                {lair && (
                  <i
                    aria-hidden
                    lang="el"
                    className="inline-grid size-4 place-items-center rounded-full bg-on-panel font-serif text-[10px] leading-none font-semibold tracking-normal text-panel not-italic"
                  >
                    Μ
                  </i>
                )}
                {label}
              </span>
              <b className="font-semibold min-[420px]:text-right">{value}</b>
            </li>
          ))}
        </ul>
        <Button asChild variant="outline" className="justify-self-start">
          <Link href="/dashboard">Face the Minotaur</Link>
        </Button>
      </div>
    </section>
  );
}

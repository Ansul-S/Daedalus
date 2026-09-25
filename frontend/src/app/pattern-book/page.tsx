import type { Metadata } from "next";

import {
  KeyPointMark,
  type KeyPointState,
  MasteryHatch,
  Pips,
  type Rating,
  RatingChip,
  type Verdict,
  VerdictMark,
} from "@/components/hatching";
import { LabyrinthMark } from "@/components/labyrinth-mark";
import { Markdown } from "@/components/markdown";
import { Sheet, SheetHead, SheetSection, TitleBlock } from "@/components/sheet";
import { ThreadTimeline } from "@/components/thread";
import { Badge } from "@/components/ui/badge";
import { RAMP } from "@/lib/glyph/grids";
import { cn } from "@/lib/utils";

import { ControlsDemo, MosaicLab, TimerDemo, TracedMark } from "./demos";

export const metadata: Metadata = {
  title: "Pattern book",
  description: "The Daedalus design system: pigments, type, controls, marks, hatching and glyph art.",
};

const PIGMENTS = [
  {
    name: "Bone",
    greek: "μήλινον",
    code: "oklch 0.955 0.016 85 · #F5F0E4",
    role: "Melinum, white earth. The paper every page is written on.",
    swatch: "bg-bone text-ink",
  },
  {
    name: "Sinopia",
    greek: "σινωπίς",
    code: "oklch 0.43 0.146 26 · #8F2121",
    role: "Red earth named for the port of Sinope. The landing page's field, Ariadne's thread, anything that needs attention. On ink it lightens to #EC5B5C.",
    swatch: "bg-sinopia text-bone",
  },
  {
    name: "Ochre",
    greek: "ώχρα",
    code: "oklch 0.78 0.14 75 · #EBA941",
    role: "Sil, yellow earth. Only what is earned: XP, levels, minted coins.",
    swatch: "bg-ochre text-ink",
  },
  {
    name: "Ink",
    greek: "μέλαν",
    code: "oklch 0.20 0.014 45 · #1C1411",
    role: "Atramentum, black. Type, hairlines, hatching, and the dark theme's ground.",
    swatch: "bg-ink text-bone",
  },
];

const CONTRAST = [
  ["Ink on bone", "Reading text", "16.0 : 1"],
  ["Bone on sinopia", "The landing page's field: headline, buttons, the picture", "7.7 : 1"],
  ["Sinopia on bone", "The thread, knots, links, focus, small accent text", "7.7 : 1"],
  ["Sinopia on raised paper", "The same, on the answer box and cards", "7.0 : 1"],
  ["Sinopia-lt on ink", "The accent in the dark theme and on ink panels", "5.4 : 1"],
  ["Sinopia-lt on ink 2", "The accent on dark-theme cards", "4.6 : 1"],
  ["Ochre on ink", "What is earned, on ink panels", "8.9 : 1"],
];

const TYPE = [
  ["type-hero · display 800", "type-hero", "Practice that remembers"],
  ["type-title · display 800", "type-title", "The Wings"],
  [
    "type-q · Source Serif 4 · 22–30 px",
    "type-q max-w-[34em]",
    "Why do simple RNNs struggle with long-range dependencies, and how does scaled dot-product attention address this?",
  ],
  [
    "Body · Source Serif 4 · 17 px",
    "max-w-[62ch]",
    "You named the vanishing gradient but not what it does: distant information fades exponentially. Attention scales the dot products by 1/√dₖ, not by dₖ, so the softmax stays out of its flat regions.",
  ],
  ["type-label · JetBrains Mono · 11 px", "type-label", "Q.043 · vanishing gradient · why / how · due in 2 days"],
] as const;

const KEY_POINTS: [KeyPointState, string, string][] = [
  ["covered", "Covered", "counts in full"],
  ["partial", "Partial", "counts half"],
  ["missing", "Missing", "counts nothing"],
];

const CLAIMS: [Verdict, string, string, string][] = [
  ["supported", "Supported, with the passage it rests on", "", "±0"],
  ["contradicted", "Contradicted by a passage", "line-through decoration-thread decoration-[1.5px]", "−0.15"],
  ["unverified", "Not in these passages: unverified", "underline decoration-dotted decoration-fg-3 underline-offset-[3px]", "no penalty"],
];

const RATINGS: [Rating, string][] = [
  ["again", "below 0.4, or any contradiction"],
  ["hard", "0.4 to 0.7"],
  ["good", "0.7 to 0.9"],
  ["easy", "0.9 and above"],
];

const MARKDOWN = `Scaled dot-product attention divides the scores by $\\sqrt{d_k}$:

$$
\\operatorname{Attention}(Q, K, V) = \\operatorname{softmax}\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right) V
$$

- **Why:** for large $d_k$ the dot products grow large and push the softmax into regions with tiny gradients.
- The base model uses \`d_k = 64\`.`;

function Part({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mt-6 border-t border-line-2 pt-3.5 first:mt-0">
      <p className="type-label mb-3 text-fg-2">{title}</p>
      {children}
    </div>
  );
}

export default function PatternBookPage() {
  return (
    <Sheet>
      <SheetSection>
        <SheetHead number="Sheets 01–07" title="Pattern book" sigil="β">
          Every piece of the design system as the app draws it: the workshop that built the
          Labyrinth, in earth pigments, engraving, drafting sheets and Greek letters. Switch the
          theme at the top; each piece has a dark form.
        </SheetHead>
      </SheetSection>

      <SheetSection aria-labelledby="pigments">
        <SheetHead as="h2" id="pigments" number="Sheet 01" title="Four pigments" sigil="χ">
          The four-colour palette of ancient painters. Status never rests on colour alone:
          hatching carries it (sheet 05).
        </SheetHead>
        <div className="grid grid-cols-2 border border-line-2 lg:grid-cols-4">
          {PIGMENTS.map((pigment, i) => (
            <article
              key={pigment.name}
              className={cn(
                "flex min-w-0 flex-col border-line-2",
                i % 2 === 1 && "border-l",
                i >= 2 && "max-lg:border-t",
                i === 2 && "lg:border-l",
              )}
            >
              <div
                className={cn(
                  "flex aspect-[1/1.02] flex-col justify-between p-3.5",
                  pigment.swatch,
                )}
              >
                <span lang="el" className="font-serif text-[1.4rem] leading-[1.1] italic">
                  {pigment.greek}
                </span>
                <span className="font-mono text-[11px] leading-normal tracking-[0.05em]">
                  {pigment.code}
                </span>
              </div>
              <div className="grid content-start gap-1.5 px-3.5 pt-3.5 pb-[18px]">
                <h3 className="font-display text-[2rem] leading-[0.9] font-extrabold uppercase">
                  {pigment.name}
                </h3>
                <p className="text-small text-fg-2">{pigment.role}</p>
              </div>
            </article>
          ))}
        </div>
        <div className="mt-8 overflow-x-auto border border-line-2">
          <table className="w-full min-w-[560px] border-collapse text-small">
            <thead>
              <tr className="type-label text-left text-[10px] text-fg-2">
                <th className="border-b border-line px-3.5 py-2.5 font-medium">Pair</th>
                <th className="border-b border-line px-3.5 py-2.5 font-medium">Used for</th>
                <th className="border-b border-line px-3.5 py-2.5 font-medium">Contrast</th>
              </tr>
            </thead>
            <tbody>
              {CONTRAST.map(([pair, use, ratio]) => (
                <tr key={pair} className="border-b border-line last:border-b-0">
                  <th className="type-label px-3.5 py-2.5 text-left font-medium">{pair}</th>
                  <td className="px-3.5 py-2.5 text-fg-2">{use}</td>
                  <td className="px-3.5 py-2.5 font-mono font-semibold whitespace-nowrap tabular-nums">
                    {ratio}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-4 max-w-[62ch] text-small text-fg-2">
          Every pair clears WCAG AA for small text (4.5 : 1).
        </p>
      </SheetSection>

      <SheetSection aria-labelledby="type">
        <SheetHead as="h2" id="type" number="Sheet 02" title="Type" sigil="γ">
          Big Shoulders for titles and numbers, Source Serif 4 for reading, JetBrains Mono for
          everything the app measures and for the glyph art. All three are free (OFL) and served
          with the app; no request goes to Google.
        </SheetHead>
        <div className="grid grid-cols-1 gap-[18px] md:grid-cols-3">
          <article className="flex flex-col gap-2.5 border border-line-2 px-5 pt-[18px] pb-5">
            <p className="type-label text-fg-2">Display · titles and figures</p>
            <p className="font-display text-[clamp(2.2rem,4.2vw,3.2rem)] leading-[0.9] font-extrabold uppercase">
              Escape the labyrinth
            </p>
            <p className="type-figure text-[clamp(1.8rem,3.2vw,2.4rem)]">0.52 · 460 XP · 3:00</p>
          </article>
          <article className="flex flex-col gap-2.5 border border-line-2 px-5 pt-[18px] pb-5">
            <p className="type-label text-fg-2">Text · questions and feedback</p>
            <p className="font-serif text-[clamp(1.8rem,3.2vw,2.4rem)] leading-[1.05] font-medium">
              Why divide by √d<sub>k</sub>?
            </p>
            <p className="font-serif text-[clamp(1.25rem,2.2vw,1.6rem)]">
              <i className="text-thread">ρ</i> 0.96 · <i className="text-thread">κ</i> 0.82 ·{" "}
              <span lang="el">ΔΑΙΔΑΛΟΣ</span>
            </p>
          </article>
          <article className="flex flex-col gap-2.5 border border-line-2 px-5 pt-[18px] pb-5">
            <p className="type-label text-fg-2">Mono · labels, timers, glyph art</p>
            <p lang="el" className="font-mono text-[clamp(1.7rem,3vw,2.3rem)] leading-[0.9] tracking-[0.06em]">
              η θ σ Σ Ψ
            </p>
            <p className="type-label">Q.043 · due in 2 days · 1:42</p>
          </article>
        </div>
        <div className="mt-8 border-t border-line-2">
          {TYPE.map(([label, className, sample]) => (
            <div
              key={label}
              className="grid grid-cols-1 items-baseline gap-x-7 gap-y-2 border-b border-dotted border-line-2 py-[18px] md:grid-cols-[13rem_minmax(0,1fr)]"
            >
              <span className="type-label text-fg-2">{label}</span>
              <span className={className}>{sample}</span>
            </div>
          ))}
          <div className="grid grid-cols-1 items-baseline gap-x-7 gap-y-2 border-b border-dotted border-line-2 py-[18px] md:grid-cols-[13rem_minmax(0,1fr)]">
            <span className="type-label text-fg-2">type-figure · display 200</span>
            <span className="type-figure text-[clamp(3rem,7vw,5.5rem)]">
              <i className="font-serif text-[0.62em] font-normal text-thread">ρ</i> 0.96 ·{" "}
              <i className="font-serif text-[0.62em] font-normal text-thread">κ</i> 0.82
            </span>
          </div>
          <div className="grid grid-cols-1 items-baseline gap-x-7 gap-y-2 border-b border-dotted border-line-2 py-[18px] md:grid-cols-[13rem_minmax(0,1fr)]">
            <span className="type-label text-fg-2">Inscription · serif caps</span>
            <span lang="el" className="font-serif text-[clamp(1.4rem,3vw,2.2rem)] leading-[1.2] font-semibold tracking-[0.24em]">
              ΔΑΙΔΑΛΟΣ ΕΠΟΙΗΣΕΝ
            </span>
          </div>
        </div>
      </SheetSection>

      <SheetSection aria-labelledby="controls">
        <SheetHead as="h2" id="controls" number="Sheet 03" title="Controls" sigil="κ">
          Buttons are lettering on a drawing; chips and tabs are mono labels in hairline boxes;
          the answer box is raised paper. Square corners throughout.
        </SheetHead>
        <ControlsDemo />
        <Part title="Chips">
          <div className="flex flex-wrap items-center gap-3">
            <Badge>Why this</Badge>
            <Badge variant="solid">Due today</Badge>
            <Badge variant="thread">Contradicted</Badge>
            <Badge variant="dashed">Unverified</Badge>
            <span className="type-label text-fg-2">
              difficulty <Pips value={3} label="difficulty" />
            </span>
          </div>
        </Part>
      </SheetSection>

      <SheetSection aria-labelledby="marks">
        <SheetHead as="h2" id="marks" number="Sheet 04" title="Marks" sigil="λ">
          The mark is the Cretan labyrinth, drawn square as on the coins of Knossos. Ariadne&apos;s
          thread is the one line in the accent colour, and timers are dimension lines borrowed
          from a technical drawing.
        </SheetHead>
        <div className="grid grid-cols-1 items-start gap-[clamp(24px,4vw,56px)] md:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)]">
          <TracedMark />
          <div>
            <Part title="At small sizes">
              <div className="flex flex-wrap items-end gap-[22px]">
                {(
                  [
                    [64, "h-16 w-[60px]"],
                    [40, "h-10 w-[37px]"],
                    [28, "h-7 w-[26px]"],
                  ] as const
                ).map(([size, box]) => (
                  <span
                    key={size}
                    className="grid justify-items-center gap-1.5 font-mono text-[10px] text-fg-2"
                  >
                    <LabyrinthMark className={cn(box, "text-fg")} />
                    {size}
                  </span>
                ))}
              </div>
            </Part>
            <Part title="Ariadne's thread · the practice timeline">
              <ThreadTimeline steps={["Question", "Answer", "Verdict", "Next"]} className="mt-3.5" />
              <p className="mt-4 text-small text-fg-2">
                Links draw the thread when you point at them:{" "}
                <a href="#hatching" className="thread-link">
                  see the hatching
                </a>
                .
              </p>
            </Part>
            <Part title="Dimension line · interview mode">
              <TimerDemo />
            </Part>
          </div>
        </div>
        <Part title="Meander · section breaks on the landing page">
          <div className="meander" aria-hidden />
        </Part>
      </SheetSection>

      <SheetSection aria-labelledby="hatching">
        <SheetHead as="h2" id="hatching" number="Sheet 05" title="Hatching" sigil="δ">
          Engravers show tone with hatching. Daedalus uses the same marks for status, so nothing
          depends on colour alone and every state still reads in black and white.
        </SheetHead>
        <div className="grid grid-cols-1 border border-line-2 md:grid-cols-2">
          <section className="min-w-0 border-line-2 px-[22px] pt-5 pb-6">
            <h3 className="type-h3 mb-2.5">Mastery</h3>
            <div className="grid grid-cols-6 gap-2">
              {[0, 1, 2, 3, 4, 5].map((level) => (
                <span key={level} className="grid gap-1.5 text-center font-mono text-[10px] text-fg-2">
                  <MasteryHatch level={level} className="text-fg" />
                  {level}
                </span>
              ))}
            </div>
            <p className="mt-3 text-small text-fg-2">
              A topic&apos;s mastery: the latest score times the chance of still recalling it today,
              in six steps.
            </p>
          </section>
          <section className="min-w-0 border-line-2 px-[22px] pt-5 pb-6 max-md:border-t md:border-l">
            <h3 className="type-h3 mb-2.5">Key points</h3>
            <ul className="grid gap-2.5">
              {KEY_POINTS.map(([state, label, value]) => (
                <li key={state} className="grid grid-cols-[22px_minmax(0,1fr)_auto] items-center gap-3 text-small">
                  <KeyPointMark state={state} />
                  <span>{label}</span>
                  <span className="font-mono text-[11px] text-fg-2">{value}</span>
                </li>
              ))}
            </ul>
          </section>
          <section className="min-w-0 border-t border-line-2 px-[22px] pt-5 pb-6">
            <h3 className="type-h3 mb-2.5">Claims</h3>
            <ul className="grid gap-2.5">
              {CLAIMS.map(([verdict, label, decoration, value]) => (
                <li key={verdict} className="grid grid-cols-[22px_minmax(0,1fr)_auto] items-center gap-3 text-small">
                  <VerdictMark verdict={verdict} />
                  <span className={decoration}>{label}</span>
                  <span className="font-mono text-[11px] text-fg-2">{value}</span>
                </li>
              ))}
            </ul>
          </section>
          <section className="min-w-0 border-t border-line-2 px-[22px] pt-5 pb-6 md:border-l">
            <h3 className="type-h3 mb-2.5">Review rating</h3>
            <ul className="grid gap-2.5">
              {RATINGS.map(([rating, rule]) => (
                <li key={rating} className="grid grid-cols-[22px_minmax(0,1fr)] items-center gap-3">
                  <RatingChip rating={rating} />
                  <span>
                    <span className="font-display text-[1.05rem] font-bold tracking-[0.05em] uppercase">
                      {rating}
                    </span>{" "}
                    <span className="font-mono text-[11px] text-fg-2">{rule}</span>
                  </span>
                </li>
              ))}
            </ul>
          </section>
        </div>
      </SheetSection>

      <SheetSection aria-labelledby="glyphs">
        <SheetHead as="h2" id="glyphs" number="Sheet 06" title="Glyph mosaic" sigil="σ">
          Pictures redrawn in Greek letters, which are also the symbols of ML maths (η learning
          rate, θ parameters, σ sigmoid, Σ sum). The reveal is denoising: each step removes some
          noise until the picture settles, as a diffusion model does.
        </SheetHead>
        <MosaicLab />
        <Part title="The ramp · 13 steps from paper to ink · % of the cell inked">
          <div className="grid max-w-xl grid-cols-13 border border-line-2">
            {RAMP.map((step, i) => (
              <span key={i} className="grid min-w-0 justify-items-center gap-1.5 border-line not-first:border-l pt-2 pb-1.5">
                <span
                  className={cn(
                    "grid h-[30px] w-[calc(100%-6px)] max-w-[22px] place-items-center font-mono text-[17px] leading-none",
                    step.reversed && "bg-fg text-ground",
                  )}
                  style={{ fontWeight: step.weight }}
                >
                  {step.glyph === " " ? " " : step.glyph}
                </span>
                <small className="font-mono text-[9px] leading-none text-fg-2">
                  {Math.round(step.coverage * 100)}
                </small>
              </span>
            ))}
          </div>
        </Part>
        <ul className="mt-8 grid grid-cols-1 gap-x-7 gap-y-[18px] md:grid-cols-3">
          <li className="border-t border-line-2 pt-3 text-small text-fg-2">
            <b className="font-semibold text-fg">How it&apos;s drawn.</b> The etching&apos;s hatching is
            blurred into tone, local contrast is lifted, and a gentle S-curve clears the paper.
            Each cell takes the step whose ink matches its darkness; there is no dithering, so flat
            areas stay calm.
          </li>
          <li className="border-t border-line-2 pt-3 text-small text-fg-2">
            <b className="font-semibold text-fg">What ships.</b> No image file: grids of steps made
            offline (<code className="font-mono">make glyphs</code>), run-length encoded, drawn on
            a canvas in the page&apos;s own ink. On a dark ground the steps flip, so the picture stays
            positive.
          </li>
          <li className="border-t border-line-2 pt-3 text-small text-fg-2">
            <b className="font-semibold text-fg">Where it comes from.</b> Charles Holroyd died in
            1917, so his etching <i>Daedalus</i> (1895, British Museum 1918,0608.347) is in the
            public domain.
          </li>
        </ul>
      </SheetSection>

      <SheetSection aria-labelledby="markdown">
        <SheetHead as="h2" id="markdown" number="Sheet 07" title="Markdown" sigil="μ">
          Questions, answers and feedback are Markdown with maths: $…$ inline and $$…$$ on its own
          line, typeset by KaTeX. Raw HTML is never rendered.
        </SheetHead>
        <div className="grid grid-cols-1 items-start gap-8 lg:grid-cols-2">
          <pre className="overflow-x-auto bg-ink p-3.5 font-mono text-[12.5px] leading-[1.7] whitespace-pre-wrap text-bone">
            {MARKDOWN}
          </pre>
          <Markdown className="max-w-[62ch]">{MARKDOWN}</Markdown>
        </div>
      </SheetSection>

      <TitleBlock
        cells={[
          { label: "Drawing", value: "Pattern book" },
          { label: "Sheets", value: "01 – 07" },
          { label: "Field", value: "Sinopia" },
          { label: "Status", value: "Approved" },
        ]}
      />
    </Sheet>
  );
}

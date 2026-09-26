import { GlyphMosaic } from "@/components/glyph-mosaic";
import { SheetHead, SheetSection } from "@/components/sheet";
import type { DrawingName } from "@/lib/glyph/drawings";

// Every box has the shape of Daedalus at his bench, the one etching among the four.
const ASPECT = "57.6 / 43";
const COLS = 72;

const STEPS: {
  numeral: string;
  step: string;
  title: string;
  text: string;
  drawing?: DrawingName;
}[] = [
  {
    numeral: "Α′",
    step: "Ingest",
    title: "Read your sources",
    text: "PDFs through Docling, notebooks cell by cell, arXiv papers from their HTML. Every passage keeps its section, page or cell, so a citation points straight back to it.",
    drawing: "page",
  },
  {
    numeral: "Β′",
    step: "Generate",
    title: "Ask what's worth asking",
    text: "Seven question styles, from intuition to trade-offs. Each key point quotes its passage word for word, and a question whose quote can't be found is turned down.",
    drawing: "meander",
  },
  {
    numeral: "Γ′",
    step: "Practise",
    title: "Meet it before you forget",
    text: "Spaced repetition (FSRS) serves what is due first, then something new from your weakest topic. Interview mode gives you three minutes.",
    drawing: "labyrinth",
  },
  {
    numeral: "Δ′",
    step: "Grade",
    title: "Checked claim by claim",
    text: "Every claim is supported, contradicted or unverified against the question's own passages, with the citation. Qwen grades; gpt\u2011oss, which writes the questions, never does.",
  },
];

export function HowItWorks() {
  return (
    <SheetSection
      id="how-it-works"
      aria-labelledby="how-it-works-title"
      className="scroll-mt-24 md:scroll-mt-16"
    >
      <SheetHead
        as="h2"
        id="how-it-works-title"
        number="Four steps"
        title="How it works"
        sigil="α"
      >
        From your study material to a graded answer. Every question and every verdict points back
        to the passages it rests on.
      </SheetHead>
      <ol className="grid grid-cols-1 gap-[22px] min-[560px]:grid-cols-2 min-[980px]:grid-cols-4">
        {STEPS.map(({ numeral, step, title, text, drawing }) => (
          <li key={step} className="grid min-w-0 content-start gap-2.5">
            {drawing ? (
              <GlyphMosaic
                drawing={drawing}
                cols={COLS}
                aspect={ASPECT}
                boxClassName="border border-line-2 text-fg"
              />
            ) : (
              <GlyphMosaic grid="thinker" boxClassName="border border-line-2 text-fg" />
            )}
            <p className="type-label mt-1.5 text-thread">
              <span lang="el">{numeral}</span> · {step}
            </p>
            <h3 className="type-h3">{title}</h3>
            <p className="text-small text-fg-2">{text}</p>
          </li>
        ))}
      </ol>
    </SheetSection>
  );
}

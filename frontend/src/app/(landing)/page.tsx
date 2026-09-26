import type { Metadata } from "next";

import { Sheet, SheetSection } from "@/components/sheet";

import { Faq } from "./faq";
import { HEADLINE, Hero } from "./hero";
import { HowItWorks } from "./how-it-works";
import { Measured } from "./measured";
import { MinotaurBand } from "./minotaur-band";

const TITLE = `Daedalus · ${HEADLINE}`;
const DESCRIPTION =
  "AI/ML interview questions written from your own PDFs, notebooks and arXiv papers, asked again just before you would forget them, and every answer checked claim by claim against the passages its question came from.";

export const metadata: Metadata = {
  title: { absolute: TITLE },
  description: DESCRIPTION,
  openGraph: { type: "website", siteName: "Daedalus", title: TITLE, description: DESCRIPTION },
  twitter: { card: "summary_large_image", title: TITLE, description: DESCRIPTION },
};

// The front page: the app's pages start at /practice.
export default function LandingPage() {
  return (
    <>
      <Sheet>
        <Hero />
        <HowItWorks />
        <SheetSection className="grid grid-cols-1 gap-[clamp(40px,6vw,72px)]">
          <MinotaurBand />
          <Measured />
          <Faq />
        </SheetSection>
      </Sheet>
      {/* the Greek key, over the colophon */}
      <div
        aria-hidden
        className="meander mx-auto mt-[clamp(40px,6vw,72px)] h-7 max-w-[1240px] [mask-size:42px_28px] [-webkit-mask-size:42px_28px]"
      />
    </>
  );
}

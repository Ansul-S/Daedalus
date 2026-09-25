import type { Metadata } from "next";
import { Suspense } from "react";

import { Sheet, SheetHead, SheetSection } from "@/components/sheet";

import { QuestionBank } from "./bank";

export const metadata: Metadata = { title: "Questions" };

export default function QuestionsPage() {
  const head = (
    <SheetHead number="Sheet Q-01" title="Questions" sigil="ζ">
      Every question written from your material: the ones practice draws from, the ones retired
      by hand and the ones the checks turned down. Open one to read its key points and the checks
      it went through, to correct it, or to rate it.
    </SheetHead>
  );
  return (
    <Sheet>
      {/* The filters are read from the address, which is only known in the browser */}
      <Suspense
        fallback={
          <SheetSection>
            {head}
            <p className="text-fg-2">Opening the question bank…</p>
          </SheetSection>
        }
      >
        <QuestionBank>{head}</QuestionBank>
      </Suspense>
    </Sheet>
  );
}

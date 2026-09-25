import type { Metadata } from "next";

import { Sheet, SheetHead } from "@/components/sheet";

import { PracticeSession } from "./session";

export const metadata: Metadata = { title: "Practice" };

export default function PracticePage() {
  return (
    <Sheet>
      <PracticeSession>
        <SheetHead number="Sheet P-01" title="Practice" sigil="π">
          One question at a time, in the order your memory needs them: what is due for review
          first, then something new from your weakest topic.
        </SheetHead>
      </PracticeSession>
    </Sheet>
  );
}

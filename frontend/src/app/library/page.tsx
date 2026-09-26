import type { Metadata } from "next";

import { Sheet, SheetHead } from "@/components/sheet";

import { Library } from "./library";

export const metadata: Metadata = { title: "Library" };

export default function LibraryPage() {
  return (
    <Sheet>
      <Library>
        <SheetHead number="Sheet L-01" title="Library" sigil="θ">
          The material every question is written from: your PDFs, notebooks and arXiv papers. Add
          a source and the worker reads it into passages; the topic map sorts them by concept; then
          questions are written from the passages nothing has asked about yet.
        </SheetHead>
      </Library>
    </Sheet>
  );
}

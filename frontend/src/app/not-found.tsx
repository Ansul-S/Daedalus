import type { Metadata } from "next";
import Link from "next/link";

import { LabyrinthMark } from "@/components/labyrinth-mark";
import { Sheet, SheetHead, SheetSection } from "@/components/sheet";
import { Button } from "@/components/ui/button";

export const metadata: Metadata = { title: "Not found" };

export default function NotFound() {
  return (
    <Sheet>
      <SheetSection className="grid grid-cols-1 gap-10 md:grid-cols-[minmax(0,1fr)_auto] md:items-start">
        <div>
          <SheetHead number="Sheet 404" title="A dead end" sigil="ω">
            No page lives at this address. Follow the thread back out.
          </SheetHead>
          <Button asChild variant="outline" className="md:ml-[calc(9.5rem+2rem)]">
            <Link href="/practice">Back to practice</Link>
          </Button>
        </div>
        <LabyrinthMark thread className="h-auto w-40 md:w-56" />
      </SheetSection>
    </Sheet>
  );
}

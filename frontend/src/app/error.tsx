"use client";

import { Sheet, SheetHead, SheetSection } from "@/components/sheet";
import { Button } from "@/components/ui/button";

export default function ErrorPage({ error, reset }: { error: Error; reset: () => void }) {
  return (
    <Sheet>
      <SheetSection>
        <SheetHead number="Sheet E-01" title="Something broke" sigil="ε">
          This page failed to draw: {error.message || "an unexpected error"}.
        </SheetHead>
        <Button variant="outline" className="md:ml-[calc(9.5rem+2rem)]" onClick={reset}>
          Try again
        </Button>
      </SheetSection>
    </Sheet>
  );
}

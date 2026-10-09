"use client";

import { useState } from "react";

import { Button } from "@/components/ui/button";

/** A button that deletes, asking again first, since a deletion can't be undone. The question
 * goes away once the deletion has gone through, and stays while it fails. */
export function DeleteButton({
  label,
  deleting,
  onDelete,
}: {
  label: string;
  deleting: boolean;
  onDelete: () => Promise<unknown>;
}) {
  const [asking, setAsking] = useState(false);

  if (!asking) {
    return (
      <div>
        <Button variant="destructive" size="sm" onClick={() => setAsking(true)}>
          {label}
        </Button>
      </div>
    );
  }
  return (
    <div className="grid gap-3 border border-l-[3px] border-line-2 border-l-thread px-[18px] py-3.5">
      <p className="font-medium">This can&apos;t be undone.</p>
      <div className="flex flex-wrap items-center gap-3">
        <Button
          variant="destructive"
          size="sm"
          disabled={deleting}
          onClick={() =>
            void onDelete().then(
              () => setAsking(false),
              () => {
                // the page says what went wrong
              },
            )
          }
        >
          {deleting ? "Deleting…" : "Delete it"}
        </Button>
        <Button variant="outline" size="sm" disabled={deleting} onClick={() => setAsking(false)}>
          Keep it
        </Button>
      </div>
    </div>
  );
}

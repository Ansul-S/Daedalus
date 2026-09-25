import * as React from "react";

import { cn } from "@/lib/utils";

// The answer sheet: raised paper under a hairline, the reading serif.
function Textarea({ className, ...props }: React.ComponentProps<"textarea">) {
  return (
    <textarea
      data-slot="textarea"
      className={cn(
        "flex field-sizing-content min-h-32 w-full border border-line-2 bg-surface px-4 py-3.5 font-serif text-base leading-[1.62] text-fg transition-colors placeholder:text-fg-3 focus-visible:border-fg disabled:cursor-not-allowed disabled:opacity-50 aria-invalid:border-thread",
        className,
      )}
      {...props}
    />
  );
}

export { Textarea };

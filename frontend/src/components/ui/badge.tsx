import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { Slot } from "radix-ui";

import { cn } from "@/lib/utils";

// A chip: a mono label in a hairline box.
const badgeVariants = cva(
  "inline-flex w-fit shrink-0 items-center gap-1 overflow-hidden border px-[7px] py-[5px] font-mono text-[10px] leading-none font-medium tracking-[0.1em] whitespace-nowrap uppercase [&>svg]:pointer-events-none [&>svg]:size-3",
  {
    variants: {
      variant: {
        default: "border-line-2 text-fg",
        solid: "border-fg bg-fg text-ground",
        thread: "border-thread text-thread",
        dashed: "border-dashed border-line-2 text-fg-2",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
);

function Badge({
  className,
  variant = "default",
  asChild = false,
  ...props
}: React.ComponentProps<"span"> & VariantProps<typeof badgeVariants> & { asChild?: boolean }) {
  const Comp = asChild ? Slot.Root : "span";

  return (
    <Comp
      data-slot="badge"
      data-variant={variant}
      className={cn(badgeVariants({ variant }), className)}
      {...props}
    />
  );
}

export { Badge, badgeVariants };

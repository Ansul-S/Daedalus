import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { Slot } from "radix-ui";

import { cn } from "@/lib/utils";

// Lettering on a drawing: square, uppercase display type, a 1.5 px rule round the edge.
const buttonVariants = cva(
  "group/button inline-flex shrink-0 cursor-pointer items-center justify-center gap-3 border-[1.5px] font-display leading-none font-bold tracking-[0.05em] whitespace-nowrap uppercase transition-transform duration-150 select-none hover:-translate-y-px disabled:pointer-events-none disabled:opacity-50 [&_kbd]:border [&_kbd]:border-current [&_kbd]:px-[5px] [&_kbd]:py-[3px] [&_kbd]:text-[11px] [&_kbd]:leading-none [&_kbd]:font-medium [&_kbd]:tracking-normal [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
  {
    variants: {
      variant: {
        default: "border-fg bg-fg text-ground",
        outline: "border-current bg-transparent",
        ghost: "border-transparent bg-transparent hover:bg-surface",
        destructive: "border-thread bg-transparent text-thread",
        link: "h-auto border-0 p-0 font-serif font-normal tracking-normal normal-case underline decoration-thread decoration-2 underline-offset-[3px] hover:translate-y-0",
      },
      size: {
        default: "px-[18px] pt-[13px] pb-3 text-lg",
        sm: "px-3.5 pt-2.5 pb-[9px] text-[0.95rem]",
        icon: "size-9",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
);

function Button({
  className,
  variant = "default",
  size = "default",
  asChild = false,
  ...props
}: React.ComponentProps<"button"> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean;
  }) {
  const Comp = asChild ? Slot.Root : "button";

  return (
    <Comp
      data-slot="button"
      data-variant={variant}
      data-size={size}
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  );
}

export { Button, buttonVariants };

import { type ClassValue, clsx } from "clsx";
import { extendTailwindMerge } from "tailwind-merge";

// The type scale's font sizes (globals.css) share the text- prefix with colours; without them
// tailwind-merge would take `text-label` for a colour and drop it next to `text-fg-2`.
const twMerge = extendTailwindMerge({
  extend: {
    theme: { text: ["hero", "title", "h3", "q", "body", "small", "label"] },
  },
});

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

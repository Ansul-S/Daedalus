"use client";

import { setThemeChoice, type ThemeChoice, useThemeChoice } from "@/lib/theme";
import { cn } from "@/lib/utils";

const CHOICES: { value: ThemeChoice; label: string }[] = [
  { value: "light", label: "Light" },
  { value: "dark", label: "Dark" },
  { value: "system", label: "Auto" },
];

export function ThemeSwitch({ className }: { className?: string }) {
  const current = useThemeChoice();

  return (
    <div
      role="group"
      aria-label="Theme"
      className={cn("inline-flex items-stretch border border-line-2", className)}
    >
      <span className="hidden items-center px-2 font-mono text-[10px] leading-none tracking-[0.1em] text-fg-2 uppercase sm:inline-flex">
        Theme
      </span>
      {CHOICES.map(({ value, label }) => (
        <button
          key={value}
          type="button"
          aria-pressed={current === value}
          onClick={() => setThemeChoice(value)}
          className="min-h-[30px] cursor-pointer border-l border-line-2 px-2.5 font-mono text-[11px] leading-none tracking-[0.05em] uppercase first-of-type:border-l-0 hover:bg-surface aria-pressed:bg-fg aria-pressed:text-ground sm:first-of-type:border-l"
        >
          {label}
        </button>
      ))}
    </div>
  );
}

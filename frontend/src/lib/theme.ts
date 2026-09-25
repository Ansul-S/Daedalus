import { useSyncExternalStore } from "react";

import { THEME_KEY } from "@/lib/theme-script";

export type Theme = "light" | "dark";
export type ThemeChoice = Theme | "system";

const DARK = "(prefers-color-scheme: dark)";

function choice(): ThemeChoice {
  const theme = document.documentElement.dataset.theme;
  return theme === "light" || theme === "dark" ? theme : "system";
}

function resolved(): Theme {
  const picked = choice();
  if (picked !== "system") return picked;
  return matchMedia(DARK).matches ? "dark" : "light";
}

function subscribe(onChange: () => void) {
  const observer = new MutationObserver(onChange);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  const media = matchMedia(DARK);
  media.addEventListener("change", onChange);
  return () => {
    observer.disconnect();
    media.removeEventListener("change", onChange);
  };
}

export function setThemeChoice(next: ThemeChoice) {
  const root = document.documentElement;
  if (next === "system") delete root.dataset.theme;
  else root.dataset.theme = next;
  try {
    if (next === "system") localStorage.removeItem(THEME_KEY);
    else localStorage.setItem(THEME_KEY, next);
  } catch {
    // storage unavailable (private mode, blocked): the choice lasts until the page is closed
  }
}

/** The reader's choice; "system" on the server and until hydration. */
export function useThemeChoice(): ThemeChoice {
  return useSyncExternalStore(subscribe, choice, () => "system");
}

/** The theme on screen, whether chosen or taken from the system. */
export function useTheme(): Theme {
  return useSyncExternalStore(subscribe, resolved, () => "light");
}

/** Calls `onChange` whenever the theme on screen may have changed; returns the unsubscribe. */
export const watchTheme = subscribe;

import { useSyncExternalStore } from "react";

// Interview mode: three minutes an answer, as in an interview. The choice is kept in this
// browser and holds for every question until it is switched off.

/** Seconds; the API gives five more XP to an answer inside it. */
export const INTERVIEW_LIMIT = 180;

const KEY = "daedalus:interview";
const CHANGED = "daedalus:interview-changed";

// Where storage is blocked, the choice lasts as long as the page
let fallback = false;

function read(): boolean {
  try {
    return localStorage.getItem(KEY) === "on";
  } catch {
    return fallback;
  }
}

function subscribe(onChange: () => void) {
  const onStorage = (event: StorageEvent) => {
    if (event.key === KEY || event.key === null) onChange();
  };
  window.addEventListener("storage", onStorage);
  window.addEventListener(CHANGED, onChange);
  return () => {
    window.removeEventListener("storage", onStorage);
    window.removeEventListener(CHANGED, onChange);
  };
}

export function setInterviewMode(on: boolean): void {
  fallback = on;
  try {
    if (on) localStorage.setItem(KEY, "on");
    else localStorage.removeItem(KEY);
  } catch {
    // storage unavailable (private mode, blocked): kept in `fallback` instead
  }
  window.dispatchEvent(new Event(CHANGED));
}

/** Whether interview mode is on; off on the server and until hydration. */
export function useInterviewMode(): boolean {
  return useSyncExternalStore(subscribe, read, () => false);
}

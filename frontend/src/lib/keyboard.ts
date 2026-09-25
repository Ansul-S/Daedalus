import { useSyncExternalStore } from "react";

// The practice page's shortcuts: ⌘↵ (Ctrl+↵ off Apple keyboards) submits, N moves on.

const unchanging = () => () => {};

function apple(): boolean {
  return /Mac|iPhone|iPad|iPod/.test(navigator.platform || navigator.userAgent);
}

/** How to write the submit shortcut on this keyboard: ⌘↵ on the server and until hydration. */
export function useSubmitKeys(): { label: string; aria: string } {
  const isApple = useSyncExternalStore(unchanging, apple, () => true);
  return isApple ? { label: "⌘↵", aria: "Meta+Enter" } : { label: "Ctrl ↵", aria: "Control+Enter" };
}

export function isSubmitKey(event: { key: string; metaKey: boolean; ctrlKey: boolean }): boolean {
  return event.key === "Enter" && (event.metaKey || event.ctrlKey);
}

/** Whether a key press belongs to a text field rather than to the page. */
export function isTyping(target: EventTarget | null): boolean {
  return (
    target instanceof HTMLElement &&
    (target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName))
  );
}

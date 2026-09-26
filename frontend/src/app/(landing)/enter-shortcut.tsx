"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";

/** Enter goes where the page's first button goes, while nothing on the page has focus. */
export function EnterShortcut({ href }: { href: string }) {
  const router = useRouter();

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key !== "Enter" || event.defaultPrevented || event.isComposing) return;
      if (event.metaKey || event.ctrlKey || event.altKey || event.shiftKey) return;
      const focused = document.activeElement;
      if (focused && focused !== document.body) return;
      router.push(href);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [router, href]);

  return null;
}

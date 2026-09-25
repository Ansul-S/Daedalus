import { useCallback, useEffect, useRef, useState } from "react";

/** Time spent on an answer: counts while `running` and the page is in view, carrying on from
 * `initial` (a saved draft's time). `seconds` ticks for display; `read()` is exact. With
 * `whileHidden` (interview mode) it counts on in another tab too, as an interviewer's clock
 * would. */
export function useStopwatch(initial: number, running: boolean, whileHidden = false) {
  const [seconds, setSeconds] = useState(Math.floor(initial));
  const counted = useRef(initial); // seconds before the current stretch
  const since = useRef<number | null>(null); // when the current stretch began

  const read = useCallback(
    () => counted.current + (since.current === null ? 0 : (performance.now() - since.current) / 1000),
    [],
  );

  useEffect(() => {
    if (!running) return;
    const start = () => {
      if (since.current === null && (whileHidden || !document.hidden)) {
        since.current = performance.now();
      }
    };
    const stop = () => {
      if (since.current === null) return;
      counted.current = read();
      since.current = null;
    };
    const onVisibility = () => (document.hidden && !whileHidden ? stop() : start());

    start();
    const tick = setInterval(() => setSeconds(Math.floor(read())), 250);
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      clearInterval(tick);
      document.removeEventListener("visibilitychange", onVisibility);
      stop();
    };
  }, [running, whileHidden, read]);

  return { seconds, read };
}

/** Whole seconds since the component mounted, whether or not the page is in view. */
export function useElapsed(): number {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const start = performance.now();
    const tick = setInterval(() => setSeconds(Math.floor((performance.now() - start) / 1000)), 250);
    return () => clearInterval(tick);
  }, []);
  return seconds;
}

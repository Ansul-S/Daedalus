// A new level: Greek letters thrown up in ochre, the colour of what is earned. The confetti
// code loads only when there is something to celebrate, and nothing moves for a reader who
// asks for less motion.

const LETTERS = ["Δ", "Α", "Ι", "Λ", "Ο", "Σ"];
const OCHRE = "#eba941";

export async function celebrate(): Promise<void> {
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  try {
    const { default: confetti } = await import("canvas-confetti");
    // The reading serif, which has the Greek letters; loaded before they are drawn
    const family =
      getComputedStyle(document.documentElement).getPropertyValue("--font-source-serif").trim() ||
      "serif";
    await document.fonts.load(`600 40px ${family}`, LETTERS.join(""));
    const scalar = 3.4;
    const shapes = LETTERS.map((text) =>
      confetti.shapeFromText({ text, scalar, color: OCHRE, fontFamily: family }),
    );
    const burst = (x: number, angle: number) =>
      confetti({
        particleCount: 26,
        angle,
        spread: 64,
        startVelocity: 52,
        origin: { x, y: 0.75 },
        shapes,
        scalar,
        ticks: 240,
        disableForReducedMotion: true,
      });
    await Promise.all([burst(0.12, 62), burst(0.88, 118)]);
  } catch {
    // no canvas, or the code didn't load: the verdict still says the level was reached
  }
}

// How a question is named on screen.

const STYLES: Record<string, string> = {
  intuition: "intuition",
  why_how: "why / how",
  compare: "compare",
  tradeoffs: "trade-offs",
  failure_modes: "failure modes",
  connection: "connection",
  paper: "paper",
};

export function styleLabel(style: string): string {
  return STYLES[style] ?? style.replaceAll("_", " ");
}

/** Q.043 */
export function questionNumber(id: number): string {
  return `Q.${String(id).padStart(3, "0")}`;
}

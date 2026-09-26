import { graderMeasurement } from "@/lib/measured";

/** The grader's agreement with hand grades, as the design notes record it, and what it cost. */
export function Measured() {
  const { spearman, kappa, answers, method } = graderMeasurement();
  const figures = [
    {
      symbol: "ρ",
      value: spearman,
      caption: "Spearman ρ between the grader's scores and hand grades",
    },
    {
      symbol: "κ",
      value: kappa,
      caption: "Cohen's κ on the covered, partial and missing labels",
    },
    { value: answers, caption: "answers graded by hand, prompt-injection attempts included" },
    { value: "$0", caption: "spent: open-source software and free tiers only" },
  ];

  return (
    <section
      aria-labelledby="measured-title"
      className="bg-panel p-[clamp(28px,4vw,44px)] text-on-panel [--focus:var(--sinopia-lt)]"
    >
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-2">
        <h2 id="measured-title" className="type-label text-sinopia-lt">
          The grader, measured
        </h2>
        <a
          href={method}
          target="_blank"
          rel="noreferrer"
          className="type-label text-on-panel/72 underline decoration-on-panel/40 underline-offset-[3px] hover:text-on-panel"
        >
          How it was measured
          <span aria-hidden> ↗</span>
          <span className="sr-only"> (opens in a new tab)</span>
        </a>
      </div>
      <ul className="mt-[18px] grid grid-cols-2 gap-6 min-[900px]:grid-cols-4">
        {figures.map(({ symbol, value, caption }) => (
          <li key={caption} className="min-w-0">
            <span className="type-figure block text-[clamp(3.2rem,7vw,5.6rem)]">
              {symbol && (
                <i className="mr-[0.08em] font-serif text-[0.6em] leading-none font-normal text-sinopia-lt">
                  {symbol}
                </i>
              )}
              {value}
            </span>
            <span className="mt-2.5 block max-w-[26ch] text-small text-on-panel/72">
              {caption}
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}

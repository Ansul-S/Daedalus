// What people ask before they trust a grader with their answers. Each answer says what the code
// does: the grading and writing chains in backend/app/llm/models.py, the score in
// backend/app/grading/scoring.py.
const QUESTIONS = [
  {
    question: "Where does my answer go?",
    answer:
      "To the model that grades it: Qwen on Groq's free tier first, then a local model, then Gemini, whose free tier uses prompts to improve Google's products. It goes with the question and the passages the question was written from. Your files stay on your machine; only passages from them are sent, to write questions and to grade answers.",
  },
  {
    question: "What if I say something the sources don't cover?",
    answer:
      "It is marked unverified, not wrong, and costs nothing. Only a claim that a passage contradicts costs points: 0.15 each, taken off what the key points earn.",
  },
  {
    question: "Who writes the questions, and who grades them?",
    answer:
      "gpt-oss writes them from your passages, on Groq's free tier (Gemini or a local model when Groq can't), and never grades the answers to them: Qwen does.",
  },
];

export function Faq() {
  return (
    <section aria-labelledby="faq-title" className="max-w-[60rem]">
      <h2 id="faq-title" className="type-label mb-3 text-fg-2">
        Asked before starting
      </h2>
      <div className="border-t border-line-2">
        {QUESTIONS.map(({ question, answer }, i) => (
          <details key={question} open={i === 0} className="group border-b border-line-2">
            <summary className="flex cursor-pointer list-none items-baseline justify-between gap-[18px] py-4 font-display text-[1.45rem] leading-[1.1] font-bold tracking-[0.02em] uppercase [&::-webkit-details-marker]:hidden">
              {question}
              <span aria-hidden className="shrink-0 font-mono text-xs leading-none text-thread">
                <span className="group-open:hidden">[+]</span>
                <span className="hidden group-open:inline">[−]</span>
              </span>
            </summary>
            <p className="mb-[18px] max-w-[62ch] text-fg-2">{answer}</p>
          </details>
        ))}
      </div>
    </section>
  );
}

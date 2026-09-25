import ReactMarkdown, { type Components } from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";

import { cn } from "@/lib/utils";

// Links in questions, answers and feedback open beside the app.
const COMPONENTS: Components = {
  a: ({ href, title, children }) => (
    <a href={href} title={title} target="_blank" rel="noreferrer" className="thread-link">
      {children}
    </a>
  ),
};

// Run into a line of text (a claim, a key point), Markdown keeps its maths, code and emphasis
// but not its blocks.
const BLOCKS = [
  "p", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "blockquote", "pre", "hr",
  "table", "thead", "tbody", "tr", "th", "td",
];

/** Questions, answers and feedback: Markdown with $maths$ and $$display maths$$. Raw HTML is
 * never rendered, since the text comes from models and from the reader. */
export function Markdown({
  children,
  inline = false,
  className,
}: {
  children: string;
  inline?: boolean;
  className?: string;
}) {
  const Wrapper = inline ? "span" : "div";
  return (
    <Wrapper className={cn("markdown", className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={COMPONENTS}
        disallowedElements={inline ? BLOCKS : undefined}
        unwrapDisallowed={inline}
      >
        {children}
      </ReactMarkdown>
    </Wrapper>
  );
}

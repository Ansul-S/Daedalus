import Link from "next/link";

import { TitleBlock } from "@/components/sheet";
import { API_URL } from "@/lib/api";

// The colophon: signed the way Greek potters signed their vases, over the drawing's title block.
export function SiteFooter() {
  return (
    <footer className="mx-auto mt-[clamp(40px,6vw,72px)] w-full max-w-[calc(1240px+2*var(--gutter))] px-(--gutter) pb-[72px]">
      <p
        lang="el"
        className="text-center font-serif text-[clamp(1.1rem,2.6vw,1.75rem)] leading-[1.1] font-semibold tracking-[0.28em]"
      >
        ΔΑΙΔΑΛΟΣ ΕΠΟΙΗΣΕΝ
      </p>
      <p className="mt-1.5 mb-6 text-center text-small text-fg-2">
        &ldquo;Daedalus made it&rdquo;, the way Greek potters signed their vases.
      </p>
      <TitleBlock
        className="border-x-0"
        cells={[
          { label: "Project", value: "Daedalus" },
          { label: "Drawing", value: "Interview practice" },
          {
            label: "Reference",
            value: (
              <Link href="/pattern-book" className="thread-link">
                Pattern book
              </Link>
            ),
          },
          {
            label: "API",
            value: (
              <a href={`${API_URL}/docs`} className="thread-link">
                Docs
              </a>
            ),
          },
          { label: "Cost", value: "$0 · free tiers" },
        ]}
      />
    </footer>
  );
}

import { cn } from "@/lib/utils";

// What a source may be shown under, as its licence URL names it: arXiv records a paper's
// licence (backend/app/ingest/arxiv.py), and a Creative Commons one asks for it to be named
// wherever the text is shown.
const CREATIVE_COMMONS =
  /creativecommons\.org\/(licenses|publicdomain)\/([a-z-]+)\/(\d+(?:\.\d+)?)/i;

/** "CC BY 4.0" for http://creativecommons.org/licenses/by/4.0/, and so on. */
export function licenceLabel(url: string): string {
  const found = CREATIVE_COMMONS.exec(url);
  if (found) {
    const [, kind, terms, version] = found;
    return kind.toLowerCase() === "publicdomain" && terms.toLowerCase() === "zero"
      ? `CC0 ${version}`
      : `CC ${terms.toUpperCase()} ${version}`;
  }
  if (/arxiv\.org\/licenses/i.test(url)) return "arXiv licence";
  return "Licence";
}

/** The licence, named and linked to its terms. */
export function Licence({ url, className }: { url: string; className?: string }) {
  return (
    <a
      href={url}
      target="_blank"
      rel="noreferrer license"
      title="The licence this source is shared under"
      className={cn(
        "thread-link font-mono text-[11px] tracking-[0.03em] whitespace-nowrap",
        className,
      )}
    >
      {licenceLabel(url)}
    </a>
  );
}

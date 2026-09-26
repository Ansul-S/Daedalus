import type { Metadata } from "next";
import { connection } from "next/server";

import { dependencies } from "@/client/sdk.gen";
import type { Check } from "@/client/types.gen";
import { Commands, QUICK_START } from "@/components/commands";
import { GlyphMosaic } from "@/components/glyph-mosaic";
import { Sheet, SheetHead, SheetSection } from "@/components/sheet";
import { SERVER_API_URL } from "@/lib/api";
import { cn } from "@/lib/utils";

export const metadata: Metadata = { title: "Setup" };

// Read on the server: this page asks the API directly, whether or not the browser can.
async function getChecks(): Promise<Check[] | null> {
  const { data } = await dependencies({ baseUrl: SERVER_API_URL });
  return data ?? null;
}

const STATUS: Record<Check["status"], { mark: string; word: string }> = {
  ok: { mark: "hatch-5", word: "ok" },
  warn: { mark: "hatch-partial", word: "warn" },
  fail: { mark: "border-thread", word: "fail" },
};

function Checks({ checks }: { checks: Check[] }) {
  return (
    <ul className="border-t border-line-2">
      {checks.map((check) => {
        const { mark, word } = STATUS[check.status];
        return (
          <li
            key={check.name}
            className="grid grid-cols-[14px_3.25rem_minmax(0,1fr)] items-baseline gap-x-3 gap-y-1 border-b border-dotted border-line-2 py-3 md:grid-cols-[14px_3.25rem_minmax(0,15rem)_minmax(0,1fr)]"
          >
            <i
              aria-hidden
              className={cn("inline-block size-3.5 translate-y-0.5 border-[1.5px] border-current", mark)}
            />
            <span className={cn("type-label", check.status === "fail" && "text-thread")}>{word}</span>
            <span className="font-medium">{check.name}</span>
            <span className="col-start-3 text-small text-fg-2 md:col-start-4">{check.detail}</span>
          </li>
        );
      })}
    </ul>
  );
}

export default async function SetupPage() {
  await connection(); // checked on every request, never at build time
  const checks = await getChecks();

  return (
    <Sheet>
      <SheetSection>
        <SheetHead number="Sheet S-01" title="Setup" sigil="ε">
          What Daedalus runs on: the database, the local models and the cloud API keys. They are
          checked each time this page loads.
        </SheetHead>
        <div className="grid grid-cols-1 gap-x-14 gap-y-10 lg:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]">
          <div className="grid min-w-0 grid-cols-1 content-start gap-8">
            {checks === null ? (
              <p className="max-w-[62ch] border border-l-[3px] border-line-2 border-l-thread px-[18px] py-3.5 text-small">
                <span className="type-label mb-1.5 block text-thread">API not reachable</span>
                Can&apos;t reach the API at <code className="font-mono">{SERVER_API_URL}</code>.
                Start it with <code className="font-mono">make api</code> and reload this page.
              </p>
            ) : (
              <Checks checks={checks} />
            )}
            <Commands
              title="Quick start · local"
              note="free tiers only"
              lines={QUICK_START}
            />
          </div>
          <GlyphMosaic
            grid="thinker"
            className="self-start text-fg"
            caption={
              <>
                Fig. 1 · Charles Holroyd, <i>Daedalus</i>, etching, 1895
              </>
            }
          />
        </div>
      </SheetSection>
    </Sheet>
  );
}

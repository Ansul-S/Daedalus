import type { Metadata } from "next";

import { Sheet, SheetHead, SheetSection } from "@/components/sheet";
import { SIGN_IN, SIGN_IN_NEEDED } from "@/lib/sign-in";

import { AccountDeleted, DeleteAccount } from "./delete-account";
import { DeletePractice } from "./delete-practice";

export const metadata: Metadata = { title: "Privacy" };

// Each line says what the code does: sign-in in src/lib/auth.ts, the grading chain in
// backend/app/llm/models.py, tracing in backend/app/llm/tracing.py, drafts in src/lib/drafts.ts,
// deleting in backend/app/api/practice.py and account.py.

// Built for production, the app is the demo, whose database is on Neon
const DATABASE = SIGN_IN_NEEDED
  ? "the demo's database, Postgres on Neon in the United States (Ohio)"
  : "this copy of the app's own database";

function Part({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="grid grid-cols-1 gap-x-8 gap-y-2 border-t border-line-2 py-6 md:grid-cols-[9.5rem_minmax(0,1fr)]">
      <h2 className="type-label text-fg-2">{title}</h2>
      <div className="grid max-w-[62ch] gap-4">{children}</div>
    </section>
  );
}

export default function PrivacyPage() {
  return (
    <Sheet>
      <SheetSection>
        <SheetHead number="Sheet N-01" title="Privacy" sigil="λ">
          What Daedalus keeps about you, where your answers go, and how to delete it.
        </SheetHead>
        <div className="border-b border-line-2">
          <Part title="Signing in">
            {SIGN_IN ? (
              <p>
                Signing in with GitHub keeps your GitHub id, your name and the link to your
                avatar, until you delete your account. GitHub is asked for public data only, so
                your email address is never seen: the account is given an address made of your
                GitHub id and username at users.noreply.github.com, which reaches no inbox.
                GitHub&apos;s tokens and your IP address are not stored.
              </p>
            ) : (
              <p>
                Sign-in is not set up here: all practice belongs to this copy of the app&apos;s
                built-in user.
              </p>
            )}
          </Part>
          <Part title="Your answers">
            <p>
              An answer goes to the model that grades it, with the question and the passages the
              question was written from: Qwen on Groq&apos;s free tier, or Gemini when Groq
              can&apos;t (run locally, a local model comes between them). Each handles it under
              its own terms, and Gemini&apos;s free tier uses prompts to improve Google&apos;s
              products.
            </p>
            <p>
              The answer, its grade, your review schedule and your ratings are kept in {DATABASE},
              so that practice picks up where you left it. Nobody else sees them.
            </p>
          </Part>
          <Part title="In this browser">
            <p>
              A draft of an answer stays in this browser until it is graded, and so do your
              choice of theme and of interview mode.
              {SIGN_IN ? " The only cookie is the one that keeps you signed in." : " No cookie is set."}
            </p>
          </Part>
          <Part title="Tracing">
            <p>
              When tracing is on, as it is for the demo, each grade is traced to Langfuse: how
              long it took, the tokens it used and which model answered. The question, your
              answer and the grade&apos;s text are left out.
            </p>
          </Part>
          <Part title="Nothing else">
            <p>
              No analytics and no advertising.
              {SIGN_IN_NEEDED && " Vercel, which serves the demo, sees each request as any web host does."}
            </p>
          </Part>
          <Part title="Deleting">
            <AccountDeleted />
            <DeletePractice />
            <DeleteAccount />
          </Part>
        </div>
      </SheetSection>
    </Sheet>
  );
}

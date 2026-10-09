"use client";

import Link from "next/link";
import { useState, useSyncExternalStore } from "react";

import { Button } from "@/components/ui/button";
import { ApiError } from "@/lib/api-errors";
import { authClient, signIn, signOut } from "@/lib/session";
import { SIGN_IN, SIGN_IN_NEEDED } from "@/lib/sign-in";

const HEADER_BUTTON =
  "cursor-pointer font-mono text-[10.5px] leading-none tracking-[0.1em] whitespace-nowrap uppercase hover:text-fg-2 disabled:opacity-50";

type Session = Pick<ReturnType<typeof authClient.useSession>, "data" | "isPending">;

// Better Auth's store of the session, the one its useSession reads
const sessionStore: { get(): Session; listen(onChange: () => void): () => void } =
  authClient.$store.atoms.session;

// As the server renders it: the session is only looked up in the browser
const LOOKING_UP: Session = { data: null, isPending: true };

function subscribe(onChange: () => void): () => void {
  return sessionStore.listen(onChange);
}

// Who is signed in. While the page hydrates it is still being looked up, as the server rendered
// it, even if the lookup has answered in the meantime; the answer follows at once. Better Auth's
// useSession gives React the answer as the server's snapshot, so a slow page whose hydration
// waits for its code would no longer match the server's HTML.
function useSession(): Session {
  return useSyncExternalStore(subscribe, () => sessionStore.get(), () => LOOKING_UP);
}

/** Whether the API refused a request for want of a signed-in visitor. */
export function needsSignIn(error: unknown): boolean {
  return error instanceof ApiError && error.status === 401;
}

/** Whether this visitor's practice waits for them to sign in, before it is asked for: where
 * practice is only for a signed-in visitor and nobody is. Undefined while the sign-in is
 * looked up (the same lookup as the header's), so a page asks once this is false and never
 * sends a request the API would refuse. */
export function useSignInFirst(): boolean | undefined {
  const { data, isPending } = useSession();
  if (!SIGN_IN_NEEDED) return false;
  // Nobody can sign in where sign-in is not set up
  if (!SIGN_IN) return true;
  return isPending ? undefined : !data;
}

/** Who is signed in, and the way in or out; nothing while sign-in is off. */
export function Account() {
  const { data, isPending } = useSession();
  const [leaving, setLeaving] = useState(false);
  if (!SIGN_IN || isPending) return null;

  if (!data) {
    return (
      <button type="button" className={HEADER_BUTTON} onClick={() => void signIn()}>
        Sign in
      </button>
    );
  }
  return (
    <p className="flex items-center gap-x-3 font-mono text-[10.5px] leading-none tracking-[0.1em] whitespace-nowrap uppercase">
      <span className="hidden max-w-[22ch] truncate text-fg-2 sm:inline">{data.user.name}</span>
      <button
        type="button"
        className={HEADER_BUTTON}
        disabled={leaving}
        onClick={() => {
          setLeaving(true);
          void signOut();
        }}
      >
        Sign out
      </button>
    </p>
  );
}

/** In place of a page's data, when the API wants a signed-in visitor for it. Titled unless
 * what it sits in says so already. */
export function SignInNeeded({ titled = true }: { titled?: boolean }) {
  return (
    <div className="max-w-[62ch]">
      {titled && <p className="mb-3 type-label text-fg-2">Signed out</p>}
      {SIGN_IN ? (
        <>
          <p className="mb-4">
            Each visitor&apos;s practice is their own: the answers, the schedule, the thread and
            the coins. Sign in with GitHub to practise and to keep them. Only your public
            profile is read, never your email.
          </p>
          <div className="flex flex-wrap items-center gap-x-5 gap-y-3">
            <Button size="sm" onClick={() => void signIn()}>
              Sign in with GitHub
            </Button>
            <Link href="/privacy" className="thread-link text-small">
              What is kept
            </Link>
          </div>
        </>
      ) : (
        <p>Practice needs a signed-in visitor, and sign-in is not set up here.</p>
      )}
    </div>
  );
}

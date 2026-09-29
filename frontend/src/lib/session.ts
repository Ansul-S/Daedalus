import { createAuthClient } from "better-auth/react";

import { AUTH_PATH, SIGN_IN } from "@/lib/sign-in";

// Signing in and out in the browser, through Better Auth in this app (src/lib/auth.ts), and the
// token that tells the API who is asking.

export const authClient = createAuthClient({ basePath: AUTH_PATH });

// Asked for again this long before it expires (it lasts 15 minutes)
const RENEW_EARLY = 60_000;
// While signed out, whether another tab has signed in is looked at again this often
const SIGNED_OUT_FOR = 5 * 60_000;

let pending: Promise<string | undefined> | null = null;
let renewAt = 0;

/** The API's token for the visitor signed in, or nothing while nobody is. Every request in
 * the meantime shares one token, and a new one is asked for shortly before it expires. */
export function apiToken(): Promise<string | undefined> {
  if (!SIGN_IN || typeof window === "undefined") return Promise.resolve(undefined);
  if (pending === null || Date.now() >= renewAt) {
    // Until this one comes, every request waits for it
    renewAt = Infinity;
    pending = fetchToken().then(
      (token) => {
        renewAt = token ? expiry(token) - RENEW_EARLY : Date.now() + SIGNED_OUT_FOR;
        return token;
      },
      (error: unknown) => {
        pending = null;
        throw error;
      },
    );
  }
  return pending;
}

// The session, and with it a new token (Better Auth's `set-auth-jwt` header): one request,
// which answers without an error while nobody is signed in.
async function fetchToken(): Promise<string | undefined> {
  let token: string | undefined;
  const { data, error } = await authClient.getSession({
    fetchOptions: {
      onSuccess: ({ response }) => {
        token = response.headers.get("set-auth-jwt") ?? undefined;
      },
    },
  });
  if (error) throw new Error(error.message ?? "The sign-in could not be checked");
  return data ? token : undefined;
}

// When a token expires, in milliseconds: its `exp` claim
function expiry(token: string): number {
  const payload = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
  const { exp } = JSON.parse(atob(payload)) as { exp?: number };
  return typeof exp === "number" ? exp * 1000 : Date.now();
}

/** Off to GitHub, and back to this page signed in. */
export async function signIn(): Promise<void> {
  await authClient.signIn.social({ provider: "github", callbackURL: window.location.pathname });
}

/** Signed out, the page starts over: nothing of the visitor's practice stays on it. */
export async function signOut(): Promise<void> {
  await authClient.signOut();
  window.location.reload();
}

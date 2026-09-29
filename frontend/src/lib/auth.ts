import { betterAuth } from "better-auth";
import { jwt } from "better-auth/plugins";
import { Pool } from "pg";

import { AUTH_PATH, SIGN_IN } from "@/lib/sign-in";

// Sign-in with GitHub, run by Better Auth on the server side of this app. Its users, sessions
// and keys live in the app's own Postgres, in the auth_* tables (migration 0010). Once a
// visitor is signed in, the browser asks for a short-lived token and sends it to the API, which
// checks it against the public keys published at /auth/jwks (backend/app/api/users.py).
//
// Kept about a visitor: their GitHub id, name and avatar link. GitHub is asked for public data
// only, so no email address is ever seen; nor are sessions' IP addresses or GitHub's tokens
// stored.

// Better Auth's address: the issuer and audience of its tokens, as the API expects them
const BASE_URL = process.env.BETTER_AUTH_URL?.replace(/\/+$/, "");

// The API's connection string names its driver (postgresql+psycopg://); node-postgres takes
// the plain scheme. The default is the API's own, docker-compose's database.
const DATABASE_URL = (
  process.env.DATABASE_URL ?? "postgresql://daedalus:daedalus@localhost:5433/daedalus"
).replace(/^postgresql\+\w+:/, "postgresql:");

function createAuth() {
  return betterAuth({
    baseURL: BASE_URL,
    basePath: AUTH_PATH,
    secret: process.env.BETTER_AUTH_SECRET,
    database: new Pool({ connectionString: DATABASE_URL }),
    // Named apart from the API's `users`, which says whose practice is whose
    user: { modelName: "auth_user" },
    session: { modelName: "auth_session" },
    account: { modelName: "auth_account", updateAccountOnSignIn: false },
    verification: { modelName: "auth_verification" },
    socialProviders: {
      github: {
        clientId: process.env.GITHUB_CLIENT_ID ?? "",
        clientSecret: process.env.GITHUB_CLIENT_SECRET ?? "",
        // No scope: public data only. Better Auth needs an email for every user, so each
        // gets GitHub's no-reply address for their account, which reaches no inbox.
        disableDefaultScope: true,
        mapProfileToUser: (profile) => ({
          email: `${profile.id}+${profile.login}@users.noreply.github.com`,
          emailVerified: false,
        }),
      },
    },
    databaseHooks: {
      // The address is still used to limit sign-in attempts; it just isn't written down.
      session: { create: { before: async (session) => ({ data: { ...session, ipAddress: null } }) } },
      // Nothing here calls GitHub after sign-in, so its tokens are not kept.
      account: {
        create: {
          before: async (account) => ({
            data: {
              ...account,
              accessToken: null,
              refreshToken: null,
              idToken: null,
              accessTokenExpiresAt: null,
              refreshTokenExpiresAt: null,
            },
          }),
        },
      },
    },
    plugins: [
      jwt({
        schema: { jwks: { modelName: "auth_jwks" } },
        jwt: {
          issuer: BASE_URL,
          audience: BASE_URL,
          // The token says who (its subject, the user's id) and a name to call them by.
          definePayload: ({ user }) => ({ name: user.name }),
        },
      }),
    ],
    telemetry: { enabled: false },
  });
}

/** Better Auth, or null when sign-in is off (src/lib/sign-in.ts). */
export const auth = SIGN_IN ? createAuth() : null;

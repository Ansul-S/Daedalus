import path from "node:path";

import type { NextConfig } from "next";

// One .env for the whole repository, at its root, the one the API reads. Variables already set
// win, as they do for the API; a deployment has no such file and takes its settings from the
// host.
try {
  process.loadEnvFile(path.join(__dirname, "..", ".env"));
} catch (error) {
  if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
}

// Sign-in with GitHub (src/lib/auth.ts) is offered once everything it needs is set; without it
// the app works as before, as the built-in user. Decided when the app is built.
const SIGN_IN_SETTINGS = [
  "BETTER_AUTH_URL",
  "BETTER_AUTH_SECRET",
  "GITHUB_CLIENT_ID",
  "GITHUB_CLIENT_SECRET",
];

const nextConfig: NextConfig = {
  env: {
    NEXT_PUBLIC_SIGN_IN: SIGN_IN_SETTINGS.every((name) => process.env[name]) ? "on" : "",
    // In production the API answers a request from nobody signed in with 401 (locally it is
    // the built-in user's), so there the pages wait for a sign-in before asking for practice.
    NEXT_PUBLIC_SIGN_IN_NEEDED: process.env.ENVIRONMENT === "production" ? "on" : "",
  },
};

export default nextConfig;

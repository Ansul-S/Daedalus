// Whether visitors can sign in: next.config.ts turns it on when the app is built with every
// setting sign-in needs. Off, the app works as the built-in user, as it did before sign-in.
export const SIGN_IN = process.env.NEXT_PUBLIC_SIGN_IN === "on";

// Where Better Auth answers, in this app. Not /api/auth: deployed, /api is the API's.
export const AUTH_PATH = "/auth";

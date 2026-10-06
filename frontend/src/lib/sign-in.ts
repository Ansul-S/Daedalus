// Whether visitors can sign in: next.config.ts turns it on when the app is built with every
// setting sign-in needs. Off, the app works as the built-in user, as it did before sign-in.
export const SIGN_IN = process.env.NEXT_PUBLIC_SIGN_IN === "on";

// Whether practice is only for a signed-in visitor: built in production (next.config.ts),
// where the API takes nobody signed in as nobody. Locally that is the built-in user.
export const SIGN_IN_NEEDED = process.env.NEXT_PUBLIC_SIGN_IN_NEEDED === "on";

// Where Better Auth answers, in this app. Not /api/auth: deployed, /api is the API's.
export const AUTH_PATH = "/auth";

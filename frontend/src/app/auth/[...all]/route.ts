import { toNextJsHandler } from "better-auth/next-js";

import { auth } from "@/lib/auth";

// Better Auth's endpoints: signing in with GitHub and its callback, the session, signing out,
// the API's token and the keys that check it. Nothing is here while sign-in is off.
const handlers = auth ? toNextJsHandler(auth) : null;

function off(): Response {
  return new Response("Sign-in is not set up", { status: 404 });
}

export function GET(request: Request): Promise<Response> | Response {
  return handlers ? handlers.GET(request) : off();
}

export function POST(request: Request): Promise<Response> | Response {
  return handlers ? handlers.POST(request) : off();
}

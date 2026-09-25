import type { CreateClientConfig } from "@/client/client.gen";

// The browser calls the API directly rather than through a Next.js proxy, whose timeout would
// cut off a grade that waits on a model. Inlined at build time.
export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// Server components may reach the API at another address (API_URL), e.g. inside a container.
export const SERVER_API_URL = process.env.API_URL ?? API_URL;

export const createClientConfig: CreateClientConfig = (config) => ({
  ...config,
  baseUrl: API_URL,
});

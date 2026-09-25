import { defineConfig } from "@hey-api/openapi-ts";

// `make client` writes openapi.json from the backend and runs this: the generated client in
// src/client/ is committed, so it changes only when the API does.
export default defineConfig({
  input: "./openapi.json",
  output: "src/client",
  plugins: [
    { name: "@hey-api/client-fetch", runtimeConfigPath: "./src/lib/api" },
    "@hey-api/typescript",
    "@hey-api/sdk",
    { name: "@tanstack/react-query", queryOptions: true, mutationOptions: true },
  ],
});

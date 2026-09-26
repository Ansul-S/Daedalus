import path from "node:path";

import { defineConfig, devices } from "@playwright/test";

// The end-to-end test: the whole app on fake models (FAKE_MODELS), in a database and a data
// folder of its own. `make e2e` makes both, checks that the ports below are free, and runs
// this; the servers are started here and stopped when the test ends.

const database = process.env.E2E_DATABASE_URL;
const dataDir = process.env.E2E_DATA_DIR;
if (!database || !dataDir) {
  throw new Error("Run the end-to-end test with `make e2e`, which makes the database it runs on.");
}

// The usual addresses: the frontend is built with the API's address in it
const API = "http://localhost:8000";
const WEB = "http://localhost:3000";
const BACKEND = path.join(__dirname, "..", "backend");

// The API and the worker
const backendEnv = {
  DATABASE_URL: database,
  DATA_DIR: dataDir,
  FAKE_MODELS: "true",
  ENVIRONMENT: "local",
  CORS_ORIGINS: JSON.stringify([WEB]),
  // The test's notebook has three short sections, which at the usual passage sizes would be
  // read as one passage
  CHUNK_MIN_TOKENS: "20",
  // Every real model out of reach, so that a call the stand-ins missed fails at once rather
  // than reaching Ollama or spending a key's quota. Set here, these beat the ones in `.env`.
  OLLAMA_BASE_URL: "http://127.0.0.1:9",
  GROQ_API_KEY: "none",
  GEMINI_API_KEY: "none",
  // The worker's output as it prints it: Playwright waits for it to say it is ready
  PYTHONUNBUFFERED: "1",
};

export default defineConfig({
  testDir: "e2e",
  // One walk through the app that waits on the worker: well under a minute, never forever
  timeout: 120_000,
  expect: { timeout: 10_000 },
  // The servers and the database are shared: one test at a time
  workers: 1,
  reporter: [["list", { printSteps: true }]],
  use: {
    ...devices["Desktop Chrome"],
    baseURL: WEB,
    contextOptions: { reducedMotion: "reduce" },
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  webServer: [
    {
      name: "API",
      command: "uv run uvicorn app.main:app --port 8000",
      cwd: BACKEND,
      url: `${API}/health`,
      env: backendEnv,
      reuseExistingServer: false,
      gracefulShutdown: { signal: "SIGINT", timeout: 5_000 },
    },
    {
      name: "Worker",
      command: "uv run --group ingest python -m scripts.worker",
      cwd: BACKEND,
      // It has no address to wait on: it says when it is ready
      wait: { stdout: /Waiting for jobs/ },
      env: backendEnv,
      stdout: "pipe",
      gracefulShutdown: { signal: "SIGINT", timeout: 5_000 },
    },
    {
      name: "Web",
      command: "pnpm build && pnpm start",
      url: WEB,
      env: { NEXT_PUBLIC_API_URL: API },
      reuseExistingServer: false,
      timeout: 300_000,
      gracefulShutdown: { signal: "SIGTERM", timeout: 5_000 },
    },
  ],
});

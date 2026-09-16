import { connection } from "next/server";

type Check = {
  name: string;
  status: "ok" | "warn" | "fail";
  detail: string;
};

// Server-only: the browser never calls the API directly from this page.
const API_URL = process.env.API_URL ?? "http://localhost:8000";

const STATUS_STYLES: Record<Check["status"], string> = {
  ok: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300",
  warn: "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300",
  fail: "bg-rose-100 text-rose-800 dark:bg-rose-900/40 dark:text-rose-300",
};

async function getChecks(): Promise<Check[] | null> {
  try {
    const response = await fetch(`${API_URL}/health/deps`);
    return response.ok ? await response.json() : null;
  } catch {
    return null;
  }
}

export default async function Home() {
  await connection(); // check status on every request, not at build time
  const checks = await getChecks();

  return (
    <main className="mx-auto w-full max-w-2xl flex-1 px-6 py-16">
      <h1 className="text-3xl font-semibold tracking-tight">Daedalus</h1>
      <p className="mt-2 text-zinc-600 dark:text-zinc-400">
        Interview questions generated from your study material, graded against the same
        sources.
      </p>

      <h2 className="mt-10 text-lg font-medium">Setup status</h2>
      {checks === null ? (
        <p className="mt-3 rounded-md border border-rose-200 bg-rose-50 p-4 text-rose-800 dark:border-rose-900 dark:bg-rose-950/40 dark:text-rose-300">
          Can&apos;t reach the API at {API_URL}. Start it with{" "}
          <code className="font-mono">make api</code>.
        </p>
      ) : (
        <ul className="mt-3 divide-y divide-zinc-200 rounded-md border border-zinc-200 dark:divide-zinc-800 dark:border-zinc-800">
          {checks.map((check) => (
            <li key={check.name} className="flex flex-wrap items-center gap-x-3 gap-y-1 px-4 py-3">
              <span
                className={`rounded px-2 py-0.5 text-xs font-medium uppercase ${STATUS_STYLES[check.status]}`}
              >
                {check.status}
              </span>
              <span className="font-medium">{check.name}</span>
              <span className="text-sm text-zinc-600 sm:ml-auto dark:text-zinc-400">
                {check.detail}
              </span>
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}

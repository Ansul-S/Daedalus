# Deploying

How to deploy a copy of Daedalus on free tiers, as the demo at https://daedalus-demo.vercel.app
is, and the checks every push passes before it deploys.

**Contents:** [Steps](#steps) · [What changes in production](#what-changes-in-production) ·
[Continuous integration](#continuous-integration)

The demo is one Vercel project (Hobby) with a Neon database (Free). `vercel.json` describes the project:
- the frontend (`frontend/`) and the API (`backend/`, `app.main:app`, up to 300 s a request, without its tests and scripts) as two services;
- `/api/...` sent to the API and everything else to the frontend;
- every function in `cle1`, beside the database;
- builds for `main` only.

## Steps

1. **The database.** Create a Neon project in the region next to the functions (aws us-east-2 for `cle1`), and cap its compute, e.g. at 0.25 CU, so that the free hours last the month. Create the tables by running `make migrate` with `DATABASE_URL` set to Neon's pooled connection string, written `postgresql+psycopg://…`.
2. **The library.** The deployed app can't ingest material or write questions, so build the library locally, in a database of its own (see [Generating questions](generating-questions.md#the-reviewed-library)). Copy its seven tables (`documents`, `chunks`, `chunk_tags`, `topics`, `chunk_topics`, `questions`, `question_sources`) to Neon keeping their ids, since questions point at their passages by id. Then move each table's id sequence past the copied rows. Keep to sources whose licence allows it: the demo's are CC BY 4.0 papers and notes written for it.
3. **Sign-in.** Create a GitHub OAuth app for the deployed address: homepage `https://<domain>`, callback `https://<domain>/auth/callback/github`.
4. **The project.** Import the repository in Vercel's dashboard; it finds the services in `vercel.json`. Set these Production environment variables (`env.example` lists them too):

   | Setting | Value |
   |---|---|
   | `ENVIRONMENT` | `production` |
   | `DATABASE_URL` | Neon's pooled connection string, written `postgresql+psycopg://…` |
   | `ROOT_PATH` | `/api` |
   | `NEXT_PUBLIC_API_URL` | `/api`: the browser's address for the API, written into the build |
   | `API_URL`, `SITE_URL` | `https://<domain>/api` (absolute: the setup page asks from the server), `https://<domain>` |
   | `BETTER_AUTH_URL`, `BETTER_AUTH_SECRET` | `https://<domain>`, and a new secret |
   | `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET` | the OAuth app's |
   | `GROQ_API_KEY`, `GEMINI_API_KEY` | the cloud models' keys |
   | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | optional: tracing, without text |

   Left unset, the daily limits hold grading to 10 a user and 80 in all. Practice days are counted in UTC unless `PRACTICE_TIMEZONE` says otherwise.
5. **Deploys.** Every push to `main` builds. Under the project's Deployment Checks, choose CI's Backend, Frontend and End-to-end jobs, and the address moves to a new deployment only once all three have passed. A push to any other branch builds nothing. A changed environment variable takes effect at the next deployment (Redeploy, in the dashboard).
6. **Checking it.** `https://<domain>/api/health/deps` shows the database and its schema's revision, whether each key is set (never its value), tracing and the daily limits.

## What changes in production

With `ENVIRONMENT=production` the API:
- grades with cloud models only (Qwen on Groq, then Gemini) and searches by keyword only;
- refuses ingestion, the topic map, generation and corrections (403);
- answers practice only to someone signed in (401 otherwise);
- holds grading to the daily limits;
- sends each piece of work's traces as it ends.

The functions sleep when nobody uses them: the first call after an idle spell took 4.4 to 4.7 s from India, then 0.5 to 1 s ([measurements](design.md#phase-6-measurements)).

## Continuous integration

GitHub Actions (`.github/workflows/`) checks every push and every pull request to `main`, in
about two minutes, with no secrets and nothing sent to a model provider:

- **Backend:** Ruff, then the tests against a Postgres service, then the grader's stand-in
  suite, whose report goes into the job summary. Every dependency group is installed except
  torch and the CUDA packages, which no test needs.
- **Frontend:** ESLint and `pnpm build`.
- **End-to-end:** `make e2e` in headless Chromium, on stand-in models. A failed walk uploads
  Playwright's trace.

The demo deploys from `main` once all three jobs pass (see [step 5](#steps)).

**Grader, live** runs only when started by hand from the Actions tab. It grades the stand-ins
with the real grader (about 16K Groq tokens, 11 minutes) and needs a `GROQ_API_KEY` repository secret.
Without one, it says so. Dependabot proposes updates to `backend/uv.lock` and to the actions
weekly. The frontend's packages are updated by hand (`pnpm update`), since pnpm's 7-day rule
refuses the way Dependabot installs them.

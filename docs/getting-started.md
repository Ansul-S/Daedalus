# Getting started

How to run Daedalus on your own Mac, add your study material and practise. Everything here is
free: open-source models through Ollama, and the free tiers of Groq and Gemini.

**Contents:** [Prerequisites](#prerequisites-macos) · [Setup](#setup) ·
[Adding study material](#adding-study-material) · [Writing questions](#writing-questions) ·
[Practising](#practising) · [Commands](#commands) · [Configuration](#configuration) ·
[Dependency safety](#dependency-safety)

## Prerequisites (macOS)

- [uv](https://docs.astral.sh/uv/), [Docker Desktop](https://www.docker.com/products/docker-desktop/), Node.js 22+
- [Ollama](https://ollama.com) and [pnpm](https://pnpm.io): `brew install ollama pnpm`

## Setup

1. **Configuration.** `cp env.example .env`, then add free API keys (optional, but recommended: question generation and grading use them first):
   - Groq: https://console.groq.com/keys
   - Google AI Studio: https://aistudio.google.com/apikey (free-tier prompts are used to improve Google's products)

   Every other setting has a working default; see [Configuration](#configuration).
2. **Database.** `make db-up` starts Postgres 17 + pgvector on **port 5433**, so it doesn't clash with a local Postgres on 5432. Then `make migrate` creates the tables.
3. **Local models.** In a separate terminal run `make ollama`. It starts Ollama with settings sized for a 16 GB Mac: 16K context, one model loaded at a time, flash attention and an 8-bit KV cache. Then download the models (about 18 GB):
   ```sh
   for m in qwen3.5:9b qwen3.5:4b gemma4:12b qwen3-embedding:0.6b; do ollama pull "$m"; done
   ```
4. **Dependencies.**
   - Backend: `cd backend && uv sync --group ingest`. The `ingest` group adds the parsing libraries (Docling, PyTorch; the environment takes about 1.3 GB), which the deployed API doesn't need.
   - Frontend: `cd frontend && pnpm install --frozen-lockfile`.

   **Always sync the backend with `uv sync --group ingest`.** A plain `uv sync` removes the ingestion libraries again, because uv removes every package that the requested dependency groups don't include. `make ingest`, `make worker` and `make test-slow` reinstall them before they run.
5. **Run.** `make api` and `make web` (each in its own terminal), then open http://localhost:3000.
   - `make web` runs the frontend's development server, which reloads as files change. To run the production build instead: `cd frontend && pnpm build && pnpm start`.
   - The app opens on its landing page, and Enter the labyrinth goes to the practice page. The Setup page, http://localhost:3000/setup, shows the status of every dependency.
   - The API's interactive documentation is at http://localhost:8000/docs.

## Adding study material

```sh
make ingest SRC="data/notes.pdf data/notebooks 1706.03762"
```

`SRC` takes files (`.pdf`, `.ipynb`), folders (searched recursively; hidden folders such as `.ipynb_checkpoints` are skipped) and arXiv IDs or URLs (`1706.03762`, `arXiv:1706.03762v7`, `https://arxiv.org/abs/1706.03762`). Relative paths may start from the repository root. Options go inside `SRC`, for example `make ingest SRC="--force data/notes.pdf"`:

| Option | Effect |
|---|---|
| `--force` | Ingest again even if the document is already ingested, e.g. after changing the chunk settings |
| `--ocr` | Recognize text in scanned PDF pages |
| `--no-formulas` | Don't convert PDF equations to LaTeX (faster) |
| `--verbose` | Show library log messages and progress bars |

- **First run.** The first PDF downloads Docling's layout, table and formula models (about 1.1 GB) into the Hugging Face cache. The first ingestion of any kind downloads the embedding model's tokenizer (11 MB).
- **Duplicates.** A file is identified by its content (SHA-256) and a paper by its arXiv ID. Adding the same material again does nothing unless `--force` is given or a different arXiv version is requested.
- **Material added through the API** is processed by the worker. Run `make worker` and leave it running; Ctrl+C stops it, and an interrupted job is queued again the next time it starts.
- **One ingestion at a time.** Only one process ingests at a time, because two copies of Docling next to the local models don't fit in 16 GB. While a worker is running, `make ingest` queues its jobs and waits for the worker to finish them.

### How ingestion works

1. **Queue.** The API or `make ingest` stores the file as `data/uploads/<sha256>.pdf` (or `.ipynb`), or records the arXiv ID, and adds a job to the `jobs` table in Postgres.
2. **Parse.** The worker claims the job.
   - PDFs go through Docling: layout, reading order, tables, and formulas as LaTeX.
   - arXiv papers are read from their HTML version on arxiv.org, or from the PDF when there is none. Title, authors and license come from arXiv's OAI-PMH interface.
   - Notebooks are read with nbformat: Markdown cells become text, code cells become code blocks, and short text outputs are kept.
3. **Chunk.** The parsed paragraphs, lists, tables, formulas and code cells are packed into chunks of 300–800 tokens. Each keeps its heading path and page or cell numbers.
   - A new section or subsection starts a new chunk.
   - Each chunk is labeled with every section it covers, for example `3 Model Architecture > 3.3 Position-wise Feed-Forward Networks · 3.4 Embeddings and Softmax · 3.5 Positional Encoding`.
4. **Embed and store.** Each chunk is embedded with `qwen3-embedding`, together with the document title and its section label, and stored with a full-text index. A document's chunks are replaced in one transaction, so a failed re-ingestion keeps the previous ones.
5. **Search.** `GET /search` takes the top 50 chunks from vector similarity and merges them, with Reciprocal Rank Fusion, with the top 10 from full-text search at half weight (see [Measuring search](grading-and-evaluation.md#measuring-search)).

Downloads and uploads are kept under `data/` (gitignored; `DATA_DIR` moves it):

```
data/
├── uploads/<sha256>.pdf, .ipynb   each file stored once, named by its content hash
├── arxiv/<id>/                    metadata.json, and v<N>.html (ar5iv.html from the fallback
│                                  site) or v<N>.pdf; v<N>.no-html records a version without HTML
├── calibration/                   answers.toml, your hand-graded answers, and grades.jsonl,
│                                  the grader's grades of them (see make calibrate)
├── eval/                          hand-questions.toml and also-answers.toml, the labels
│                                  the retrieval report adds (see make eval-retrieval)
└── reports/                       retrieval-<date>.md and grader-<date>.md, the reports of
                                   make eval-retrieval and make eval-grader
```

## Writing questions

With material ingested, write questions from it. The quickest way is a batch: build the topic
map, then write questions from the library page (with `make worker` running) or the command line.

```sh
make topics             # once, after ingesting
make generate N=20      # write 20 questions
```

For a library you will keep, list the ideas each document explains, and read each question
before it goes in. Both ways are described in [Generating questions](generating-questions.md).

## Practising

With the API running and the frontend built and started (step 5 of [Setup](#setup)), open http://localhost:3000 and press **Enter the labyrinth**. Set `PRACTICE_TIMEZONE` to your time zone first: practice days, the streak and due dates are counted in it.

- **One question at a time.** The practice page serves the question due for review, most overdue first; with nothing due, a new question from your weakest topic; with nothing new, the one you are likeliest to have forgotten. It says why it picked it.
- **Answer** in Markdown, with `$…$` for LaTeX, and press `⌘↵`. The verdict comes back in about 2 s: each key point covered, partly covered or missing, each claim checked against a cited passage, the score and the rating it earns, and the day the question comes back, 1 to 30 practice days later (see [Grading answers](grading-and-evaluation.md#grading-answers), and the [API](api.md) on the schedule).
- **Interview mode** gives each answer three minutes, counted down on a dimension line that runs on into overtime.
- **Progress.** Answers earn XP, levels, a daily streak and coins. The dashboard draws your topics as a labyrinth of rooms, hatched by mastery, with the Minotaur in the weakest.
- **The question bank** (`/questions`) shows every question with its passages and checks, and corrects, retires or rates it. **The library** (`/library`) adds material, builds the topic map and writes questions, with `make worker` running.
- **Signing in** is optional locally: without it, all practice belongs to a built-in user. With a GitHub OAuth app's id and secret, `BETTER_AUTH_URL` and a `BETTER_AUTH_SECRET` in `.env` (see `env.example`), the header offers sign-in with GitHub, and each person who signs in keeps a practice of their own. GitHub is asked for the public profile only.
- **Daily limits** on grading are off locally unless `DAILY_GRADES_PER_USER` or `DAILY_GRADES` is set; deployed, they hold each visitor to 10 grades a practice day and everyone to 80 over 24 hours. The practice page shows the grades left, and past a limit it keeps the answer as a draft until grading opens again.
- **Privacy** (`/privacy`) says what is kept about you and where answers go, and deletes your practice: your answers and grades, the review schedule and your ratings, and the drafts in the browser.

Each page is described in [frontend/README.md](../frontend/README.md#pages).

## Commands

| Command | What it does |
|---|---|
| `make db-up`, `make db-down` | Starts / stops Postgres (data is kept in a Docker volume) |
| `make migrate` | Applies database migrations |
| `make ollama` | Runs Ollama with settings sized for a 16 GB Mac |
| `make api`, `make web` | Runs the API on port 8000 / the frontend's development server on port 3000 |
| `make client` | Regenerates the frontend's typed API client from the backend's routes. Run it after changing the API |
| `make glyphs DAEDALUS=… MINOTAUR=…` | Redraws the frontend's glyph pictures from copies of two etchings, Charles Holroyd's *Daedalus* and Antonio Tempesta's *Theseus and the Minotaur* (see [frontend/README.md](../frontend/README.md#generated-code)) |
| `make ingest SRC="…"` | Ingests files, folders and arXiv papers (see [Adding study material](#adding-study-material)) |
| `make topics` | Tags every chunk with the local model and clusters the tags into topics |
| `make generate N=20` | Writes questions from the topic map (see [Generating questions](generating-questions.md)) |
| `make worker` | Processes jobs queued through the API: ingestion first, then the topic map, then question batches |
| `make calibrate` | Grades hand-graded answers and measures how far the grader agrees (see [Checking the grader](grading-and-evaluation.md#checking-the-grader-against-your-own-grades)) |
| `make eval-retrieval` | Measures search against the questions' own passages and writes the retrieval report (see [Measuring search](grading-and-evaluation.md#measuring-search)) |
| `make eval-grader` | Checks the grader against hand-graded answers and fails when it no longer agrees (see [The grader's regression suite](grading-and-evaluation.md#the-graders-regression-suite)) |
| `make check` | Checks the database and its migrations, Ollama and its models, API keys, and whether model calls are traced. `make check LIVE=1` also sends a one-word prompt to each model, traces them, and says whether Langfuse took the traces. |
| `make test` | Backend tests. Database tests use a separate `daedalus_test` database and are skipped when Postgres isn't running (`make db-up`); in CI they fail instead. |
| `make test-slow` | The PDF parsing test, which loads Docling's models |
| `make e2e` | Walks through the app in a browser on stand-in models, in a database of its own: from the landing page through the library to a graded answer, the day's last grade used, its room on the dashboard, and the practice deleted. `make e2e ARGS=--headed` shows the browser. It needs Postgres (`make db-up`), ports 8000 and 3000 free, and Playwright's Chromium, downloaded once (see [frontend/README.md](../frontend/README.md#the-end-to-end-test)) |
| `make lint` | Ruff (backend) and ESLint (frontend) |
| `make help` | Lists all commands |

## Configuration

Settings come from environment variables, then from `.env`. `env.example` lists them with their defaults; an empty value means the default.

| Setting | Default | Purpose |
|---|---|---|
| `ENVIRONMENT` | `local` | `production` means a free cloud host: cloud models only, keyword search only, no ingestion |
| `CORS_ORIGINS` | `["http://localhost:3000"]` | Origins allowed to call the API |
| `DATABASE_URL` | `postgresql+psycopg://daedalus:daedalus@localhost:5433/daedalus` | Matches `docker-compose.yml` |
| `ROOT_PATH` | empty | The path a proxy serves the API under and passes on with each request: `/api` on Vercel. Routes match without it, and the API's docs ask for its schema with it |
| `BETTER_AUTH_URL`, `BETTER_AUTH_SECRET`, `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET` | empty | Sign-in with GitHub, offered once all four are set: the frontend's address (the issuer of the tokens the API checks), a long random secret (e.g. `openssl rand -base64 32`), and a GitHub OAuth app's id and secret, with the callback `<frontend>/auth/callback/github`. Without them, practice is the built-in user's |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local model server |
| `GRADER_MODEL`, `HELPER_MODEL`, `SECOND_OPINION_MODEL`, `EMBEDDING_MODEL` | `qwen3.5:9b`, `qwen3.5:4b`, `gemma4:12b`, `qwen3-embedding:0.6b` | Local models |
| `EMBEDDING_NUM_CTX` | `2048` | Context size of embedding requests. It keeps the embedding model at about 2 GB (2.9 GB at 16K) and still fits an 800-token chunk with its title and label. |
| `GROQ_API_KEY`, `GEMINI_API_KEY` | empty | Free cloud models. `GROQ_MODEL` (`openai/gpt-oss-120b`) writes questions, `GROQ_GRADING_MODEL` (`qwen/qwen3.8-27b`) grades answers, and `GEMINI_MODEL` (`gemini-3.5-flash`) stands in for either |
| `DATA_DIR` | `data/` in the repository | Uploaded files, arXiv downloads and calibration files |
| `MAX_UPLOAD_MB` | `50` | Largest accepted file |
| `CHUNK_MIN_TOKENS`, `CHUNK_MAX_TOKENS` | `300`, `800` | Chunk size range. Documents keep their chunks until they are ingested again with `--force`. |
| `TOKENIZER_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Tokenizer that measures chunk sizes: the embedding model's own |
| `TOPIC_SIMILARITY` | `0.8` | How close two concept tags have to be to share a topic. Higher keeps topics narrow; 0.7 merged RNN, LSTM, ReLU and dropout into one. |
| `DUPLICATE_SIMILARITY` | `0.75` | Above this, two questions are the same question in other words. Short texts sit much closer together than passages do, so the line is far below the 0.9 it looks like it should be. |
| `DAILY_GRADES_PER_USER`, `DAILY_GRADES` | empty | Daily limits on grading: grades a user a practice day, and grades by everyone over the last 24 hours. Empty, there is no limit locally, and 10 and 80 in production; 0 switches grading off |
| `PRACTICE_TIMEZONE` | `UTC` | The time zone practice days are counted in, e.g. `Asia/Kolkata`. A day starts at 04:00 there, so a session past midnight still counts as one day. |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | empty | A Langfuse project's keys. With both set, every model call is traced (see [Tracing model calls](models-and-tracing.md#tracing-model-calls)) |
| `LANGFUSE_BASE_URL` | `https://cloud.langfuse.com` | The Langfuse project's region: `https://us.cloud.langfuse.com` for the US, or a self-hosted address. `LANGFUSE_HOST`, the older name, is read too |
| `LANGFUSE_CONTENT` | `false` | Sends prompts, passages, answers, the models' replies and the text of errors with the traces |
| `LANGFUSE_TRACING_ENABLED` | `true` | `false` stops tracing and keeps the keys. The tests set it, so they never send a trace |
| `FAKE_MODELS` | `false` | Stand-ins for every model, for testing (`backend/app/llm/fakes.py`): they answer at once, the same way every time, and send nothing anywhere. What they write is made up, so point `DATABASE_URL` at a throwaway database. `make e2e` runs the whole app on them in a database of its own. `make check` says when they are on, `make calibrate` refuses them, and so does `ENVIRONMENT=production`. |

The frontend reads the same `.env` at the repository's root (variables already set win). It takes the sign-in settings and `DATABASE_URL`, where Better Auth keeps its tables, from there, and `ENVIRONMENT`, which decides at build time whether practice waits for a sign-in. It also has three settings of its own, which can go in the same file:

| Setting | Default | Purpose |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | The API's address as the browser sees it; the browser calls the API directly. It is written into the build, so a change needs `pnpm build` again. The API's `CORS_ORIGINS` has to include the frontend's address. |
| `API_URL` | `NEXT_PUBLIC_API_URL` | The API's address as the frontend's server sees it, for when the two differ (e.g. in a container). The Setup page checks the API from the server. |
| `SITE_URL` | `http://localhost:3000` | The frontend's own address, from which a shared link's preview picture is fetched. It is written into the build. |

## Dependency safety

- **Python:** uv ignores any package uploaded to PyPI in the last 7 days (`exclude-newer` in `backend/pyproject.toml`). Commit `uv.lock`.
- **DeepEval** (the `eval` group) caps click, rich and tabulate below the versions the API and ingestion use, and uv resolves every group together. `override-dependencies` lifts those caps and keeps the bounds every other package sets, so the evaluation tools don't change what the app runs on; the grader suite's tests cover the parts of DeepEval it uses.
- **Frontend:** pnpm only installs versions published at least 7 days ago (`minimumReleaseAge`) and blocks dependency install scripts unless they're allowed (`allowBuilds`); both are set in `frontend/pnpm-workspace.yaml`. The pnpm version itself is pinned in `package.json`. Commit `pnpm-lock.yaml`.
- **Secrets:** never commit `.env`; `.gitignore` already excludes it.

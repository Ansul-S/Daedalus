# Daedalus

AI/ML interview practice built on your own study material. Daedalus generates conceptual interview questions from your PDFs, Jupyter notebooks and arXiv papers, then grades your answers against those same sources, with citations.

Everything runs on free resources: open-source models on your Mac (Ollama), plus free cloud tiers (Groq, Gemini) for bulk work and fast grading.

**Status:** Phases 1–3 are built: ingestion and retrieval, question generation, and grading. PDFs, notebooks and arXiv papers are parsed, split into chunks, embedded and searchable with page, cell or section citations; a topic map is built over them, and questions are written from those passages and checked against them. Answers are graded against the same passages, every key point labelled and every claim checked against a cited source, and on 75 hand-graded answers the grader's scores rank them as the hand grades do (Spearman ρ 0.96). Next is Phase 4, the practice app. Architecture, decisions, measurements and roadmap: [docs/design.md](docs/design.md).

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
5. **Run.** `make api` and `make web` (each in its own terminal), then open http://localhost:3000. The page shows the status of every dependency. The API's interactive documentation is at http://localhost:8000/docs.

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
5. **Search.** `GET /search` takes the top 50 chunks from vector similarity and the top 50 from full-text search, and merges the two lists with Reciprocal Rank Fusion.

Downloads and uploads are kept under `data/` (gitignored; `DATA_DIR` moves it):

```
data/
├── uploads/<sha256>.pdf, .ipynb   each file stored once, named by its content hash
├── arxiv/<id>/                    metadata.json, and v<N>.html (ar5iv.html from the fallback
│                                  site) or v<N>.pdf; v<N>.no-html records a version without HTML
└── calibration/                   answers.toml, your hand-graded answers, and grades.jsonl,
                                   the grader's grades of them (see make calibrate)
```

## Generating questions

Questions are written from the material you have ingested, in two steps. The first reads the
library with the local model and has to run once after ingesting; the second writes questions
and can run as often as you like.

```sh
make topics             # tag every chunk and cluster the tags into topics
make generate N=20      # write 20 questions
```

### 1. The topic map

`make topics` asks `qwen3.5:4b` what each chunk explains, for two to five concept tags, and
whether the chunk is worth asking a question about. Every distinct tag is then embedded and
clustered, so the same idea in a paper and in a notebook lands in one topic.

Options go in `ARGS`, for example `make topics ARGS="--rules-only"`:

| Option | Effect |
|---|---|
| `--document ID`, `--limit N` | Tag one document only, or at most N chunks |
| `--retag` | Tag chunks again that already have tags, e.g. after changing the prompt |
| `--tag-only`, `--cluster-only` | Stop before clustering, or cluster the tags there already are |
| `--rules-only` | Judge the stored tags by the context-only rules again, with no model calls |
| `--similarity X` | How close two tags have to be to share a topic (default `TOPIC_SIMILARITY`, 0.8) |

- It takes about 12 s per chunk, so roughly 25 minutes for a library of 124 chunks. Ollama
  must be running, and only one process may use the local models at a time.
- A chunk is context-only, and never asked about, when the model says it explains nothing or
  when a rule says so: chunks that are only code, and scaffolding such as roadmaps, learning
  objectives, setup and imports, acknowledgments and front matter. Context-only chunks keep
  their tags and can still be shown alongside a question.
- Topics keep their ids as long as their names survive, so questions stay filed where they are.

### 2. Writing questions

`make generate N=20` plans the batch, writes it down, and then works through it. Each question
goes to Groq's `gpt-oss-120b` (Gemini, then the local model, if Groq is unavailable) with its
source chunks in delimiters, and comes back as JSON: the question, a reference answer, two to
four key points each carrying an exact quote from a source, misconceptions, and a difficulty.

Options go in `ARGS`, for example `make generate ARGS="--job 17"`:

| Option | Effect |
|---|---|
| `--count N` | How many questions to write (also `N=` on `make generate`) |
| `--document ID` | Ask about one document only |
| `--job ID` | Carry on with a batch that stopped early |
| `--verbose` | Show library log messages |

- **Passages are picked a source at a time and a topic at a time,** so a batch spreads over
  the whole library instead of the one document that happens to hold the most material. The
  style rotates through intuition, why/how, compare, trade-offs, failure modes, connection and
  paper questions.
- **A chunk an accepted question already covers is left out,** so a second run breaks new ground.
- **Before it runs, the whole plan is written down** as one row per question. Ctrl+C loses at
  most the question in flight; `make generate ARGS="--job 16"` carries the rest on.
- **Each provider keeps its own pace** (Groq's free tier allows 30 requests and 8,000 tokens a
  minute, 200,000 a day). A provider that is briefly full is waited out rather than abandoned,
  and the batch stops cleanly when the day's budget is gone, leaving the rest queued.
- **Questions queued through the API** are written by the worker, the same one that ingests.

### What a question has to pass

Every question is stored either way, with the report behind the verdict, so a rejected one
says why.

| Check | A question is turned down when |
|---|---|
| quotes | a key point's quote is not in the chunk it names, after one round to repair it |
| answerable | the local model, reading only the sources, cannot answer it |
| trivia | that model reads it as recalling a fact rather than explaining something |
| duplicate | it is within `DUPLICATE_SIMILARITY` of a question already accepted |

## Grading answers

Answer a question from the library, and the answer is graded against the passages the
question was written from:

```sh
curl -X POST localhost:8000/questions/31/attempts -H 'Content-Type: application/json' \
  -d '{"answer": "An RNN carries a hidden state from one step to the next, so each word is read in the light of the ones before it."}'
```

The grader labels what it sees, and the score is worked out from the labels:

| Part of a grade | What it says |
|---|---|
| key points | each one covered, partial or missing, with the words of the answer that show it |
| claims | each claim supported or contradicted by a passage, which is cited and linked the way search results are, or unverified when no passage addresses it |
| score | the key points' weighted coverage (covered 1, partial 0.5), less 0.15 for each contradicted claim, never below 0 |
| clarity | 1 to 5 for how clearly the answer is written, apart from what it says |
| feedback | strengths, gaps, errors, a model answer drawn from the passages, and a follow-up question |

- **An unverified claim costs nothing.** The passages don't cover it, which doesn't make it
  wrong.
- **Qwen 3.8 on Groq grades,** in about 2 s. The free tier holds it to 1,000 written tokens a
  minute, so a second answer within the same minute waits its turn. Without Groq the local
  `qwen3.5:9b` grades (over a minute an answer), then Gemini; gpt-oss, which writes the
  questions, never grades the answers to them. Answers are sent to Groq.
- **An answer is kept whatever happens to its grading.** When no model can grade it, it gets
  a failed grade that says why, and `POST /attempts/{id}/grades` grades it again. Every grade
  is kept.
- **The answer is untrusted text.** Instructions inside it, such as a request for full
  marks, are ignored.

### Checking the grader against your own grades

`make calibrate` measures how far the grader agrees with grades given by hand. It needs
`GROQ_API_KEY`.

```sh
make calibrate ARGS=template   # write data/calibration/answers.toml, listing every accepted question
make calibrate                 # grade the answers not graded yet, then report
make calibrate ARGS=report     # report on the grades already made, without grading
```

- **In the file,** add each answer under its question, with one label per key point, how many
  of its claims contradict the sources and, optionally, your own score out of 10. The file's
  header shows the format.
- **The report** gives Spearman's ρ between the grader's scores and the scores your labels
  give (trusted from 0.7), the same against your own scores, Cohen's κ on the key-point
  labels, where the two disagree, and the largest differences.
- **Grading uses Groq's Qwen alone,** about one answer a minute: a grade from a fallback model
  would measure that model instead. When Groq says the day's allowance is spent, grading
  stops and the next run carries on.
- **Grades are kept** in `data/calibration/grades.jsonl`, per answer and prompt version, so a
  second run grades only what is new. Both files are personal practice data and stay out of
  the repository.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health`, `GET /health/deps` | The API is up; status of the database, the models and the API keys |
| `POST /documents/upload` | Upload a `.pdf` or `.ipynb` as the multipart field `file`. Options as query parameters: `ocr`, `formulas`, `force` |
| `POST /documents/arxiv` | Add a paper: `{"arxiv_id": "1706.03762"}`, optionally with `ocr`, `formulas` and `force` |
| `GET /documents`, `GET /documents/{id}` | Documents with their status, details, chunk count and latest job |
| `GET /jobs/{id}` | A job's status, progress and error |
| `GET /search?q=…&limit=10&mode=hybrid` | Search all chunks. `mode` is `hybrid`, `vector` or `keyword`; `limit` is at most 50 |
| `POST /questions/generate` | Plan and queue a batch: `{"count": 20}`, optionally `document_id`. Returns **202** with the job the worker will run |
| `GET /questions` | The library, newest first. Filters: `status`, `topic_id`, `style`, `difficulty`, `document_id`, `source_updated`; `limit` and `offset`, with the total of the whole match |
| `GET /questions/{id}` | One question with its sources, key points and quotes, misconceptions, validation report and token usage |
| `GET /topics` | The topic map with the passages and questions behind each topic |
| `POST /questions/{id}/attempts` | Answer an accepted question: `{"answer": "…"}`, at most 8,000 characters. Returns **201** with the attempt and its grade |
| `POST /attempts/{id}/grades` | Grade an attempt again, e.g. after a failed grade. Earlier grades are kept |
| `GET /attempts/{id}` | An attempt with every grade it was given, oldest first |
| `GET /questions/{id}/attempts` | The answers given to a question, newest first, with their grades; `limit` (at most 100) and `offset` |

- **Adding material.** Both `POST` endpoints return **202** while the document's job is queued or running, and **200** when there is nothing to wait for because the document is already ingested.
  - A file over `MAX_UPLOAD_MB` gets 413; any other file type gets 415.
  - With `ENVIRONMENT=production`, both return 403, since ingestion runs locally.
- **Search results.** Each result has the chunk text, its section label, its page or cell range, and its rank in each retriever. It also has a citation, such as `RNN Intuition, pp. 7–8` or `Attention Is All You Need, § 3.2.1 Scaled Dot-Product Attention · 3.2.2 Multi-Head Attention`. For arXiv papers, a link points to the section or PDF page.
- **Without Ollama,** hybrid search falls back to keyword search and says so in `warning`, and `mode=vector` returns 503.
- **Starting a batch** needs the worker to be running, and returns the batch already in flight rather than planning a second one: two plans made at the same time would pick the same passages and pay for them twice. It returns **200** with no job when nothing is left to ask about, and 403 with `ENVIRONMENT=production`, since checking a question needs the local models.
- **A question is served with everything behind it:** the passages it was written from, cited as search results are, what an answer has to cover with the quote that proves each point, and the report from every check it went through, whether it passed or failed.
- **A grade is served with what it refers to:** each key point's text and weight next to its label, and each claim with the passage behind its verdict, cited and linked. A question that was rejected or retired can't be answered (404). Grading runs on the cloud models first, so unlike writing questions it also works with `ENVIRONMENT=production`.

```sh
curl -X POST localhost:8000/documents/arxiv -H 'Content-Type: application/json' -d '{"arxiv_id": "1706.03762"}'
curl -F file=@notes.pdf 'localhost:8000/documents/upload?formulas=false'
curl 'localhost:8000/search?q=why+scale+dot-product+attention&limit=5'
curl -X POST localhost:8000/questions/generate -H 'Content-Type: application/json' -d '{"count": 20}'
curl 'localhost:8000/questions?status=accepted&difficulty=3&limit=5'
curl 'localhost:8000/questions/31/attempts?limit=5'
```

## Commands

| Command | What it does |
|---|---|
| `make db-up`, `make db-down` | Starts / stops Postgres (data is kept in a Docker volume) |
| `make migrate` | Applies database migrations |
| `make ollama` | Runs Ollama with settings sized for a 16 GB Mac |
| `make api`, `make web` | Runs the API on port 8000 / the frontend on port 3000 |
| `make ingest SRC="…"` | Ingests files, folders and arXiv papers (see [Adding study material](#adding-study-material)) |
| `make topics` | Tags every chunk with the local model and clusters the tags into topics |
| `make generate N=20` | Writes questions from the topic map (see [Generating questions](#generating-questions)) |
| `make worker` | Processes jobs queued through the API, both ingestion and question batches |
| `make calibrate` | Grades hand-graded answers and measures how far the grader agrees (see [Checking the grader](#checking-the-grader-against-your-own-grades)) |
| `make check` | Checks the database and its migrations, Ollama and its models, and API keys. `make check LIVE=1` also sends a one-word prompt to each model. |
| `make test` | Backend tests. Database tests use a separate `daedalus_test` database and are skipped when Postgres isn't running (`make db-up`). |
| `make test-slow` | The end-to-end PDF test, which loads Docling's models |
| `make lint` | Ruff (backend) and ESLint (frontend) |
| `make help` | Lists all commands |

## Configuration

Settings come from environment variables, then from `.env`. `env.example` lists them with their defaults; an empty value means the default.

| Setting | Default | Purpose |
|---|---|---|
| `ENVIRONMENT` | `local` | `production` means a free cloud host: cloud models only, keyword search only, no ingestion |
| `CORS_ORIGINS` | `["http://localhost:3000"]` | Origins allowed to call the API |
| `DATABASE_URL` | `postgresql+psycopg://daedalus:daedalus@localhost:5433/daedalus` | Matches `docker-compose.yml` |
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

## Layout

```
backend/          FastAPI app (uv)
  app/api/        HTTP routes: health, documents and jobs, search, questions and topics,
                  attempts and grades
  app/core/       settings, setup checks
  app/db/         tables (SQLAlchemy) and Alembic migrations
  app/grading/    grading an answer against its question's sources, and scoring it
  app/ingest/     parsers (PDF, arXiv, notebooks), chunking, file storage, job queue, pipeline
  app/llm/        model routing, per-provider pacing, embeddings
  app/questions/  concept tags, topics, generation, quote grounding, validation, batch runs
  app/retrieval/  hybrid search and rank fusion
  scripts/        setup check, ingest, worker, topics, generate and calibrate commands
  tests/
frontend/         Next.js (App Router, TypeScript, Tailwind)
db/init/          SQL that runs when the database is first created (enables pgvector)
docs/             Design, decisions, measurements and roadmap
data/             Your material, uploads, downloads and calibration answers (not committed)
```

## Model routing

`backend/app/llm/models.py` decides which model does what. Pydantic AI's `FallbackModel` moves to the next model if one fails:

- **Question generation:** Groq → Gemini → local `qwen3.5:9b`
- **Grading:** Groq `qwen/qwen3.8-27b` → local `qwen3.5:9b` → Gemini. gpt-oss, which writes the questions, never grades the answers to them.
- **Tagging chunks and checking questions:** local `qwen3.5:4b` only, with no fallback. Both run over the whole library, so they stay off the cloud quotas, and both run with thinking off, temperature 0 and a fixed seed, which makes them repeatable.
- With `ENVIRONMENT=production` (free cloud hosting), Ollama is skipped and only cloud models are used.

In a batch and when grading, each cloud model also keeps its own pace: a sliding window of requests and tokens (and, for Qwen on Groq, of the tokens it writes), a daily allowance that comes back over 24 hours, and a wait when the provider says `retry-after`. A provider that is briefly full is waited out rather than abandoned, so a 429 doesn't spend the next provider's quota; one that says the day's allowance is spent is skipped until a request's worth has come back, and the next model takes over meanwhile.

## Dependency safety

- **Python:** uv ignores any package uploaded to PyPI in the last 7 days (`exclude-newer` in `backend/pyproject.toml`). Commit `uv.lock`.
- **Frontend:** pnpm only installs versions published at least 7 days ago (`minimumReleaseAge`) and blocks dependency install scripts unless they're allowed (`allowBuilds`); both are set in `frontend/pnpm-workspace.yaml`. The pnpm version itself is pinned in `package.json`. Commit `pnpm-lock.yaml`.
- **Secrets:** never commit `.env`; `.gitignore` already excludes it.

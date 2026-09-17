# Daedalus

AI/ML interview practice built on your own study material. Daedalus generates conceptual interview questions from your PDFs, Jupyter notebooks and arXiv papers, then grades your answers against those same sources, with citations.

Everything runs on free resources: open-source models on your Mac (Ollama), plus free cloud tiers (Groq, Gemini) for bulk work.

**Status:** Phase 1 (ingestion and retrieval) is complete. PDFs, notebooks and arXiv papers are parsed, split into chunks, embedded and searchable through the API, with page, cell or section citations. Next is Phase 2, question generation. Architecture, decisions, measurements and roadmap: [docs/design.md](docs/design.md).

## Prerequisites (macOS)

- [uv](https://docs.astral.sh/uv/), [Docker Desktop](https://www.docker.com/products/docker-desktop/), Node.js 22+
- [Ollama](https://ollama.com) and [pnpm](https://pnpm.io): `brew install ollama pnpm`

## Setup

1. **Configuration.** `cp env.example .env`, then add free API keys (optional, but recommended for question generation):
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
└── arxiv/<id>/                    metadata.json, and v<N>.html (ar5iv.html from the fallback
                                   site) or v<N>.pdf; v<N>.no-html records a version without HTML
```

## API

| Endpoint | Purpose |
|---|---|
| `GET /health`, `GET /health/deps` | The API is up; status of the database, the models and the API keys |
| `POST /documents/upload` | Upload a `.pdf` or `.ipynb` as the multipart field `file`. Options as query parameters: `ocr`, `formulas`, `force` |
| `POST /documents/arxiv` | Add a paper: `{"arxiv_id": "1706.03762"}`, optionally with `ocr`, `formulas` and `force` |
| `GET /documents`, `GET /documents/{id}` | Documents with their status, details, chunk count and latest job |
| `GET /jobs/{id}` | A job's status, progress and error |
| `GET /search?q=…&limit=10&mode=hybrid` | Search all chunks. `mode` is `hybrid`, `vector` or `keyword`; `limit` is at most 50 |

- **Adding material.** Both `POST` endpoints return **202** while the document's job is queued or running, and **200** when there is nothing to wait for because the document is already ingested.
  - A file over `MAX_UPLOAD_MB` gets 413; any other file type gets 415.
  - With `ENVIRONMENT=production`, both return 403, since ingestion runs locally.
- **Search results.** Each result has the chunk text, its section label, its page or cell range, and its rank in each retriever. It also has a citation, such as `RNN Intuition, pp. 7–8` or `Attention Is All You Need, § 3.2.1 Scaled Dot-Product Attention · 3.2.2 Multi-Head Attention`. For arXiv papers, a link points to the section or PDF page.
- **Without Ollama,** hybrid search falls back to keyword search and says so in `warning`, and `mode=vector` returns 503.

```sh
curl -X POST localhost:8000/documents/arxiv -H 'Content-Type: application/json' -d '{"arxiv_id": "1706.03762"}'
curl -F file=@notes.pdf 'localhost:8000/documents/upload?formulas=false'
curl 'localhost:8000/search?q=why+scale+dot-product+attention&limit=5'
```

## Commands

| Command | What it does |
|---|---|
| `make db-up`, `make db-down` | Starts / stops Postgres (data is kept in a Docker volume) |
| `make migrate` | Applies database migrations |
| `make ollama` | Runs Ollama with settings sized for a 16 GB Mac |
| `make api`, `make web` | Runs the API on port 8000 / the frontend on port 3000 |
| `make ingest SRC="…"` | Ingests files, folders and arXiv papers (see [Adding study material](#adding-study-material)) |
| `make worker` | Processes ingestion jobs queued through the API |
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
| `GROQ_API_KEY`, `GEMINI_API_KEY` | empty | Free cloud models (`GROQ_MODEL`, `GEMINI_MODEL` choose which) |
| `DATA_DIR` | `data/` in the repository | Uploaded files and arXiv downloads |
| `MAX_UPLOAD_MB` | `50` | Largest accepted file |
| `CHUNK_MIN_TOKENS`, `CHUNK_MAX_TOKENS` | `300`, `800` | Chunk size range. Documents keep their chunks until they are ingested again with `--force`. |
| `TOKENIZER_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Tokenizer that measures chunk sizes: the embedding model's own |

## Layout

```
backend/          FastAPI app (uv)
  app/api/        HTTP routes: health, documents and jobs, search
  app/core/       settings, setup checks
  app/db/         tables (SQLAlchemy) and Alembic migrations
  app/ingest/     parsers (PDF, arXiv, notebooks), chunking, file storage, job queue, pipeline
  app/llm/        model routing, embeddings
  app/retrieval/  hybrid search and rank fusion
  scripts/        setup check, ingest and worker commands
  tests/
frontend/         Next.js (App Router, TypeScript, Tailwind)
db/init/          SQL that runs when the database is first created (enables pgvector)
docs/             Design, decisions, measurements and roadmap
data/             Your material, uploads and downloads (not committed)
```

## Model routing

`backend/app/llm/models.py` decides which model does what. Pydantic AI's `FallbackModel` moves to the next model if one fails:

- **Question generation:** Groq → Gemini → local `qwen3.5:9b`
- **Grading:** local `qwen3.5:9b` → Groq → Gemini
- With `ENVIRONMENT=production` (free cloud hosting), Ollama is skipped and only cloud models are used.

## Dependency safety

- **Python:** uv ignores any package uploaded to PyPI in the last 7 days (`exclude-newer` in `backend/pyproject.toml`). Commit `uv.lock`.
- **Frontend:** pnpm only installs versions published at least 7 days ago (`minimumReleaseAge`) and blocks dependency install scripts unless they're allowed (`allowBuilds`); both are set in `frontend/pnpm-workspace.yaml`. The pnpm version itself is pinned in `package.json`. Commit `pnpm-lock.yaml`.
- **Secrets:** never commit `.env`; `.gitignore` already excludes it.

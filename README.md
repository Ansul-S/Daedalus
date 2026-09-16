# Daedalus

AI/ML interview practice built on your own study material. Daedalus generates conceptual interview questions from your PDFs, Jupyter notebooks and arXiv papers, then grades your answers against those same sources, with citations.

Everything runs on free resources: open-source models on your Mac (Ollama), plus free cloud tiers (Groq, Gemini) for bulk work.

**Status:** Phase 0 (setup) in progress. Architecture, tech stack and roadmap: [docs/design.md](docs/design.md).

## Prerequisites (macOS)

- [uv](https://docs.astral.sh/uv/), [Docker Desktop](https://www.docker.com/products/docker-desktop/), Node.js 22+
- [Ollama](https://ollama.com) and [pnpm](https://pnpm.io): `brew install ollama pnpm`

## Setup

1. **Configuration.** `cp env.example .env`, then add free API keys (optional, but recommended for question generation):
   - Groq: https://console.groq.com/keys
   - Google AI Studio: https://aistudio.google.com/apikey (free-tier prompts are used to improve Google's products)
2. **Database.** `make db-up` starts Postgres 17 + pgvector on **port 5433**, so it doesn't clash with a local Postgres on 5432.
3. **Local models.** In a separate terminal run `make ollama`. It starts Ollama with settings sized for a 16 GB Mac: 16K context, one model loaded at a time, flash attention and an 8-bit KV cache. Then download the models (about 18 GB):
   ```sh
   for m in qwen3.5:9b qwen3.5:4b gemma4:12b qwen3-embedding:0.6b; do ollama pull "$m"; done
   ```
4. **Dependencies.** `cd backend && uv sync`, then `cd frontend && pnpm install --frozen-lockfile`.
5. **Run.** `make api` and `make web` (each in its own terminal), then open http://localhost:3000. The page shows the status of every dependency.

## Commands

| Command | What it does |
|---|---|
| `make check` | Checks the database, Ollama and its models, and API keys. `make check LIVE=1` also sends a one-word prompt to each model. |
| `make test` | Backend tests |
| `make lint` | Ruff (backend) and ESLint (frontend) |
| `make db-down` | Stops Postgres (data is kept in a Docker volume) |
| `make help` | Lists all commands |

## Layout

```
backend/    FastAPI app (uv). app/core: settings + checks, app/llm: model routing, scripts/, tests/
frontend/   Next.js (App Router, TypeScript, Tailwind)
db/init/    SQL that runs when the database is first created (enables pgvector)
docs/       Design, tech stack and roadmap
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

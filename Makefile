.PHONY: help ollama db-up db-down migrate api web worker ingest check test test-slow lint

help: ## List commands
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-10s %s\n", $$1, $$2}'

ollama: ## Run the Ollama server with settings sized for a 16 GB Mac
	OLLAMA_CONTEXT_LENGTH=16384 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=5m \
	OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve

db-up: ## Start Postgres + pgvector
	docker compose up -d --wait db

db-down: ## Stop Postgres (data is kept)
	docker compose down

migrate: ## Apply database migrations
	cd backend && uv run alembic upgrade head

api: ## Run the FastAPI backend on http://localhost:8000
	cd backend && uv run uvicorn app.main:app --reload --port 8000

web: ## Run the Next.js frontend on http://localhost:3000
	cd frontend && pnpm dev

worker: ## Process ingestion jobs queued through the API (Ctrl+C to stop)
	cd backend && uv run --group ingest python -m scripts.worker

ingest: ## Ingest files, folders or arXiv IDs: make ingest SRC="data/notes.pdf 1706.03762"
	cd backend && uv run --group ingest python -m scripts.ingest $(SRC)

check: ## Check database, Ollama and API keys (LIVE=1 also sends a test prompt to each model)
	cd backend && uv run python -m scripts.check_setup $(if $(LIVE),--live,)

test: ## Run backend tests (database tests need `make db-up` and are skipped without it)
	cd backend && uv run pytest

test-slow: ## Run the slow tests that load Docling's PDF models
	cd backend && uv run --group ingest pytest -m slow

lint: ## Lint backend and frontend
	cd backend && uv run ruff check . && uv run ruff format --check .
	cd frontend && pnpm lint

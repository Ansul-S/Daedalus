.PHONY: help ollama db-up db-down migrate api web client glyphs worker ingest topics generate calibrate check test test-slow e2e lint

help: ## List commands
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-10s %s\n", $$1, $$2}'

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

client: ## Regenerate the frontend's typed API client from the backend's routes
	cd backend && uv run python -m scripts.openapi ../frontend/openapi.json
	cd frontend && pnpm exec openapi-ts

glyphs: ## Redraw the two etchings in Greek letters for the frontend: make glyphs DAEDALUS=daedalus.jpeg MINOTAUR=minotaur.jpg
	cd backend && uv run --group glyphs python -m scripts.glyphs --daedalus $(DAEDALUS) --minotaur $(MINOTAUR)

worker: ## Process jobs queued through the API: ingestion, topic map, questions (Ctrl+C to stop)
	cd backend && uv run --group ingest python -m scripts.worker

ingest: ## Ingest files, folders or arXiv IDs: make ingest SRC="data/notes.pdf 1706.03762"
	cd backend && uv run --group ingest python -m scripts.ingest $(SRC)

topics: ## Build the topic map: tag chunks with the local model, then cluster the tags
	cd backend && uv run --group ingest python -m scripts.topics $(ARGS)

generate: ## Write questions from the topic map: make generate N=20
	cd backend && uv run python -m scripts.generate $(if $(N),--count $(N),) $(ARGS)

calibrate: ## Grade hand-graded answers and measure agreement: make calibrate [ARGS=template|report]
	cd backend && uv run --group eval python -m scripts.calibrate $(ARGS)

check: ## Check database, Ollama and API keys (LIVE=1 also sends a test prompt to each model)
	cd backend && uv run python -m scripts.check_setup $(if $(LIVE),--live,)

test: ## Run backend tests (database tests need `make db-up` and are skipped without it)
	cd backend && uv run pytest

test-slow: ## Run the slow tests that load Docling's PDF models
	cd backend && uv run --group ingest pytest -m slow

e2e: ## Walk through the app in a browser on fake models, in a database of its own: make e2e [ARGS=--headed]
	cd backend && uv run python -m scripts.e2e $(ARGS)

lint: ## Lint backend and frontend
	cd backend && uv run ruff check . && uv run ruff format --check .
	cd frontend && pnpm lint

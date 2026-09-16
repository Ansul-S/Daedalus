.PHONY: help ollama db-up db-down api web check test lint

help: ## List commands
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-10s %s\n", $$1, $$2}'

ollama: ## Run the Ollama server with settings sized for a 16 GB Mac
	OLLAMA_CONTEXT_LENGTH=16384 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=5m ollama serve

db-up: ## Start Postgres + pgvector
	docker compose up -d --wait db

db-down: ## Stop Postgres (data is kept)
	docker compose down

api: ## Run the FastAPI backend on http://localhost:8000
	cd backend && uv run uvicorn app.main:app --reload --port 8000

web: ## Run the Next.js frontend on http://localhost:3000
	cd frontend && npm run dev

check: ## Check database, Ollama and API keys (LIVE=1 also sends a test prompt to each model)
	cd backend && uv run python -m scripts.check_setup $(if $(LIVE),--live,)

test: ## Run backend tests
	cd backend && uv run pytest

lint: ## Lint backend and frontend
	cd backend && uv run ruff check . && uv run ruff format --check .
	cd frontend && npm run lint

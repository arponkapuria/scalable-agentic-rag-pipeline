.PHONY: help install dev lint fmt test up down build upb stop restart down-all nuke logs ps

help:
	@echo "RAG Platform Commands:"
	@echo ""
	@echo "  --- Development ---"
	@echo "  make install    - Install Python dependencies"
	@echo "  make dev        - Run FastAPI server locally (creates .env from .env.example if missing)"
	@echo "  make lint       - Run ruff linter"
	@echo "  make fmt        - Format code with ruff"
	@echo "  make test       - Run test suite"
	@echo ""
	@echo "  --- Docker (project convention: explicit PROFILE) ---"
	@echo "  make up PROFILE=core,cache,...   - Start containers (no build)"
	@echo "  make down PROFILE=...            - Remove containers for these profiles (or all if omitted)"
	@echo ""
	@echo "  --- Convenience (personal workflow, defaults to your current phase's profiles) ---"
	@echo "  make build       - No-op (all services use stock images)"
	@echo "  make upb         - Build + start"
	@echo "  make stop        - Stop containers, KEEP volumes/data (safe, quick, resume later with 'make up')"
	@echo "  make restart     - Stop then start again (same profiles)"
	@echo "  make down-all    - Remove ALL containers, KEEP volumes (heavier than stop, still non-destructive)"
	@echo "  make logs        - Tail logs for the profiles' services"
	@echo "  make ps          - Show container status for the profiles' services"
	@echo ""
	@echo "  --- DANGER ---"
	@echo "  make nuke        - Remove ALL containers AND volumes (deletes Postgres/Qdrant/MinIO data, asks to confirm)"

# Development
install:
	uv sync --group api --group dev --group ingestion

# Create .env from .env.example if it doesn't exist yet.
.env:
	@test -f .env.example || { echo "Error: .env and .env.example are both missing."; exit 1; }
	cp .env.example .env
	@echo "Created .env from .env.example - review it, then re-run."

# Run the API locally (hot reload)
dev: .env
	uv run uvicorn services.api.main:app --reload --host 0.0.0.0 --port 8000 --env-file .env

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

test:
	uv run pytest tests/ -v


# Docker: project convention (explicit PROFILE)
# Usage: make up PROFILE=core or make up PROFILE=core,cache (See README for which profiles each phase needs)
comma := ,
empty :=
space := $(empty) $(empty)
profile_flags = $(foreach p,$(subst $(comma),$(space),$(PROFILE)),--profile $(p))

up:
ifndef PROFILE
	$(error PROFILE is required, e.g. make up PROFILE=core (see README for more details))
endif
	docker compose $(profile_flags) up -d

down:
ifndef PROFILE
	docker compose down
else
	docker compose $(profile_flags) down
endif


# Docker: convenience targets (personal workflow, not a project convention)
# Default profiles = whatever phase you're currently testing. Override with PROFILE=... same as above.
#
# PROFILE ?= is target-scoped (only these targets), NOT a global default. A global default would defeat the "PROFILE is required" check on up/down.

DEFAULT_PROFILES := core,cache,storage,vector,sandbox

build upb stop restart logs ps: PROFILE ?= $(DEFAULT_PROFILES)

# All services use stock images (postgres, redis-stack, qdrant, minio), so there is nothing to build. 
# Kept as a no-op so 'make upb' needs no special case.
build:
	@echo "Nothing to build - all services use stock images."

upb: build
	docker compose $(profile_flags) up -d

# Stop containers but keep them + volumes (fast resume via 'make up').
# Use this for "pause for the day".
stop:
	docker compose $(profile_flags) stop

restart: stop
	docker compose $(profile_flags) up -d

# Remove containers (frees more RAM/CPU than 'stop') but keep named volumes,
# so Qdrant/MinIO data survives. NOT the same as 'docker compose down -v'.
down-all:
	docker compose --profile '*' down

logs:
	docker compose $(profile_flags) logs -f --tail=100
 
ps:
	docker compose $(profile_flags) ps
 
# DESTRUCTIVE: removes ALL containers AND named volumes (Postgres, Qdrant,
# MinIO data is lost). Requires typing "yes". This is the 'down -v' reset.
nuke:
	@read -p "⚠️ This command will DELETE all containers AND volumes, including all stored data. This action cannot be UNDONE. Type yes to continue: " ans; \
	if [ "$$ans" = "yes" ]; then \
		docker compose --profile '*' down -v; \
	else \
		echo "Aborted."; exit 1; \
	fi
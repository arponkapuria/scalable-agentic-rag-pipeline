.PHONY: help install dev up down build upb stop restart down-all deploy infra test lint fmt

help:
	@echo "	 RAG Platform Commands:"
	@echo "  make install    - Install Python dependencies"
	@echo "  make dev        - Run FastAPI server locally"
	@echo ""
	@echo "  --- Docker (project convention: explicit PROFILE) ---"
	@echo "  make up PROFILE=core,cache,...   - Start containers (no build)"
	@echo "  make down PROFILE=...            - Remove containers for these profiles (or all if omitted)"
	@echo ""
	@echo "  --- Convenience (personal workflow, defaults to your current phase's profiles) ---"
	@echo "  make stop        - Stop containers, KEEP volumes/data (safe, quick, resume later with 'make up')"
	@echo "  make restart     - Stop then start again (same profiles)"
	@echo "  make down-all    - Remove ALL containers, KEEP volumes (heavier than stop, still non-destructive)"
	@echo ""
	@echo "  make infra      - Apply Terraform"
	@echo "  make deploy     - Deploy to AWS EKS via Helm"
	@echo "  make lint       - Run ruff linter"
	@echo "  make fmt        - Format code with ruff"
	@echo "  make test       - Run test suite"

# Install dependencies using uv
install:
	uv sync --group api --group dev --group ingestion

# Run Local Development Environment (Docker services)
# 8GB RAM dev box: never bring up the full stack. Usage: make up PROFILE=core
# or multiple: make up PROFILE=core,cache
# See PROJECT_INSTRUCTIONS.md's profile table for which profiles a given phase needs.
#
# ifndef/else/endif below are real Makefile conditionals (evaluated at parse
# time), so they must NOT be tab-indented -- only the actual recipe lines
# (docker compose ...) get a leading tab. Indenting the conditionals turns
# them into literal recipe text, and $(error ...) inside recipe text gets
# expanded unconditionally regardless of whether PROFILE is set -- that was
# the bug: `make up PROFILE=core` failed every time, not just when PROFILE
# was missing.
comma := ,
empty :=
space := $(empty) $(empty)
profile_flags = $(foreach p,$(subst $(comma),$(space),$(PROFILE)),--profile $(p))

up:
ifndef PROFILE
	$(error PROFILE is required, e.g. make up PROFILE=core (see PROJECT_INSTRUCTIONS.md profile table))
endif
	docker compose $(profile_flags) up -d

down:
ifndef PROFILE
	docker compose down
else
	docker compose $(profile_flags) down
endif

# --- Convenience targets below: personal workflow only, not a project convention ---
# Default profile set = whatever phase you're currently testing. Override any
# target with PROFILE=... same as the targets above.
#
# PROFILE ?= is target-scoped (only applies to build/upb/stop/restart and
# their recipes), NOT a global default -- a global default here would
# silently defeat up/down's "PROFILE is required" check above, which was a
# deliberate Phase 0 fix (bare `make up` used to boot the whole stack by
# accident). Convenience must not weaken that safety property.
DEFAULT_PROFILES := core,cache,storage,vector,sandbox

build upb stop restart: PROFILE ?= $(DEFAULT_PROFILES)

# Every service now uses a stock image (postgres, redis-stack, qdrant,
# minio) — nothing left to build locally since Ray's Dockerfile.local was
# removed along with the ray-head/ray-worker services. Kept as a no-op
# rather than deleted, so 'make upb' (build+up) doesn't need its own
# special-cased path.
build:
	@echo "Nothing to build — all services use stock images."

upb: build
	docker compose $(profile_flags) up -d

# Stops containers but leaves them + volumes in place (fast resume via
# 'make up' later, no re-registering MinIO webhook/event rules, no
# re-pulling images). This is what you want for "pause for the day."
stop:
	docker compose $(profile_flags) stop

restart: stop
	docker compose $(profile_flags) up -d

# Removes containers (frees more RAM/CPU than 'stop') but keeps named
# volumes -- Qdrant/MinIO data survives, only re-run 'make upb' to come
# back, no data loss. NOT the same as 'docker compose down -v'.
down-all:
	docker compose --profile '*' down

# Run the API locally (Hot Reload)
dev:
	uv run uvicorn services.api.main:app --reload --host 0.0.0.0 --port 8000 --env-file .env

# Infrastructure (Terraform)
infra:
	cd infra/terraform && terraform init && terraform apply

# Kubernetes Deployment (Helm)
deploy:
	# Update dependencies
	helm dependency update deploy/helm/api

	# Install/Upgrade
	helm upgrade --install api deploy/helm/api --namespace default
	helm upgrade --install ray-cluster kuberay/ray-cluster -f deploy/ray/ray-cluster.yaml

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

test:
	uv run pytest tests/ -v

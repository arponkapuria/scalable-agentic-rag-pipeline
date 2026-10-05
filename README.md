<div align="center">

# 🛠️📝 DocRAG 📝🛠️

**Cost-Aware Multi-Tenant Agentic RAG Pipeline for Document Q&A**


<!-- Core Platform -->
<img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=FFD43B" alt="Python" />
<img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
<img src="https://img.shields.io/badge/uv-DE5FE9?logo=uv&logoColor=white" alt="uv" />

<!-- AI & Orchestration -->
<img src="https://img.shields.io/badge/LangGraph-1C3C3C?logo=langgraph&logoColor=white" alt="LangGraph" />

<!-- Data Layer -->
<img src="https://img.shields.io/badge/PostgreSQL-316192?logo=postgresql&logoColor=white" alt="PostgreSQL" />
<img src="https://img.shields.io/badge/Redis-DC382D?logo=redis&logoColor=white" alt="Redis" />
<img src="https://img.shields.io/badge/Qdrant-DC244C?logo=qdrant&logoColor=white" alt="Qdrant" />
<img src="https://img.shields.io/badge/MinIO-C72E49?logo=minio&logoColor=white" />

<!-- Infrastructure -->
<img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white" />
<img src="https://img.shields.io/badge/Kubernetes-326CE5?logo=kubernetes&logoColor=white" />

<!-- Observability & Evaluation -->
<img src="https://img.shields.io/badge/Langfuse-000000?logo=clickhouse&logoColor=white" />

<br>
<br>

DocRAG is a multi-tenant, agentic RAG pipeline for document Q&A. Upload your documents, ask questions, and get grounded answers with precise citations from the source material. There is no login, and expired data is cleaned up automatically after each session ends.

Designed for zero-cost deployment using free-tier infrastructure, with a local-first development workflow.


[Overview](#overview) · [Architecture](#architecture) · [API Reference](#api-reference) · [Getting Started](#getting-started) · [Evaluation](#evaluation) · [Roadmap](#roadmap)

</div>

## Overview

- **Ingestion**: PDF, DOCX, HTML, and Markdown all go through a single Docling pipeline that preserves table structure and extracts figures instead of dumping raw text. Figures are captioned by a vision model.
- **Retrieval**: Hybrid search in Qdrant combines dense vectors with sparse BM25, merged using Reciprocal Rank Fusion and reranked by a cross-encoder.
- **Agent**: A LangGraph state machine decides, for each query, whether to retrieve, call a tool (web search or code sandbox), or answer directly. Planning, tool execution, and generation are separate nodes.
- **LLM routing**: Groq and OpenRouter work as an automatic failover pair. Ollama and vLLM backends are part of the `LLMClient` interface for local and offline use, but they haven't been validated in a live run yet.
- **Caching**: Responses are cached in Redis Stack, both exact-match and semantic-match. Cache entries are versioned, so a new upload doesn't silently invalidate old answers.
- **Isolation**: Every visitor gets a `corpus_id` from an httpOnly session cookie. Each store (Qdrant, Postgres, Redis, object storage) is filtered or tagged by it, and a background job purges everything when the session expires.
- **Guardrails**: With no login to rely on, the API uses regex-based input and output filtering plus rate limiting.

## Architecture

![Docrag Architecture](/assets/docrag-architecture.png)

**Project layout**
 
| Path | Contents |
|---|---|
| `services/api/app/` | Routes, agent, clients, caching, session handling |
| `pipelines/ingestion/` | Parsing, chunking, embedding, and indexing (runs in-process) |
| `libs/` | Shared utilities: rate limiting, circuit breakers, guardrails |
| `models/` | Reference configs for the models in use (not loaded at runtime) |
| `eval/` | RAGAS-based evaluation harness |
| `scripts/` | Dev utilities: load testing, bulk S3 upload |
| `tests/` | Unit tests |

Four storage systems, each doing one job: `Postgres` for chat history and feedback, `Qdrant` for vector search, `Redis Stack` for sessions and caching, `MinIO` for uploaded files.

## API Reference
 
Every route that touches data reads `corpus_id` from the httpOnly session cookie. It is never accepted as a request parameter, so a client can't reach another client's data by guessing an ID.
 
| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/session/init` | Issues the session cookie, or refreshes an existing valid one |
| `POST` | `/api/v1/upload/generate-presigned-url` | Returns a presigned MinIO/S3 PUT URL for direct client upload |
| `POST` | `/api/v1/webhooks/minio` | Receives MinIO bucket notifications and triggers ingestion |
| `GET` | `/api/v1/ingest/status/{file_id}` | Reports the ingestion stage of a single document |
| `GET` | `/api/v1/corpus/documents` | Lists every document in the caller's corpus |
| `POST` | `/api/v1/chat/stream` | Main chat endpoint: NDJSON stream of status events, then the final answer |
| `POST` | `/api/v1/feedback/` | Records a thumbs up/down (with an optional category) on an answer |
| `GET` | `/health/liveness` | Confirms the process is running |
| `GET` | `/health/readiness` | Checks dependency connectivity (Redis) |

## Getting Started
 
#### Prerequisites
 
- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- Docker and Docker Compose

#### 1. Clone and install
 
```bash
git clone https://github.com/arponkapuria/scalable-agentic-rag-pipeline.git
cd scalable-agentic-rag-pipeline
make install
```
 
#### 2. Configure the environment
 
```bash
cp .env.example .env
```
 
The only thing you need for a working chat path is a `GROQ_API_KEY` (there's a free tier at [console.groq.com](https://console.groq.com)). Everything else degrades gracefully if left unset: without OpenRouter you lose the failover, without Mistral ingestion still completes but figures go uncaptioned, and without Tavily web search simply reports itself as disabled.
 
#### 3. Start the services
 
The full stack is never run at once. Every service sits behind a Docker Compose profile, so you start only what the task at hand needs:
 
```bash
make up PROFILE=core,cache,storage,vector,sandbox
```
 
| Profile | Service | Used for |
|---|---|---|
| `core` | Postgres | Chat history, feedback, document tracking |
| `cache` | Redis Stack | Sessions and response caching |
| `vector` | Qdrant | Hybrid retrieval |
| `storage` | MinIO | Document uploads |
| `sandbox` | Code-exec container | The sandbox tool |
 
Managing the containers:
 
| Command | Effect |
|---|---|
| `make stop` | Pauses the default profiles; containers and volumes are kept |
| `make down PROFILE=...` | Removes the containers for those profiles; volumes are kept |
| `make down-all` | Removes all containers |
| `make nuke` | Removes everything, volumes included |
 
Pass `PROFILE=...` to `make stop` to target other profiles. Plain `docker compose down` does nothing here because every service is profile-gated.
 
#### 4. Run the API
 
```bash
make dev
```
 
The API is served at `http://localhost:8000` with the chat UI at `/`. Hot reload is on.
 
#### 5. Test and lint
 
```bash
make test   # pytest
make lint   # ruff check
make fmt    # ruff format
```
 
## Evaluation
 
The `eval/` package runs the shipped pipeline end to end against a fixed corpus and a hand-verified question set. Scoring uses the real **RAGAS** metrics (not a reimplementation) with a swappable LLM judge. It is kept separate from the `api` service's dependencies on purpose: `ragas` alone pulls in about 30 packages that have no place in a deployed container.
 
```bash
uv sync --group eval
python -m eval.cli retrieve
python -m eval.cli generate
python -m eval.cli judge --retrieval --judge-backend cohere
python -m eval.cli judge --generation --judge-backend cohere
python -m eval.cli report
```
 
Methodology, the corpus, and the question set are documented in `eval/EVALUATION.md`. Current results are in `eval/results/report.md`.
 
<!-- ## Known limitations
- No relevance-score threshold before generation yet; a weak-but-nonzero retrieval result still reaches the LLM, caught after the fact by a refusal check rather than before.
- `CACHE_SCHEMA_VERSION` invalidation is a manual bump, not tied to deploys yet.
- Sandbox has no network isolation outside Kubernetes — the Compose-local version is not egress-restricted.
- No database migration tooling; schema changes need a manual `ALTER TABLE`.
- Ollama and vLLM backends are implemented but not yet exercised in a live end-to-end run. -->
 
## Roadmap
 
- Observability pipeline
- Deployment

## License
 
Released under the [MIT License](https://opensource.org/licenses/MIT).
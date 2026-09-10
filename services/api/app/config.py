import os
from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode
from typing import Annotated, List, Optional

class Settings(BaseSettings):
    """
    Application Configuration.
    Reads environment variables automatically (case-insensitive).
    """
    # General
    ENV: str = "prod"
    LOG_LEVEL: str = "INFO"
    
    # Database (Postgres)
    DATABASE_URL: str  # e.g., postgresql+asyncpg://user:pass@host:5432/db
    
    # Redis (Cache)
    REDIS_URL: str     # e.g., redis://elasticache-endpoint:6379/0
    
    # Vector DB (Qdrant)
    QDRANT_HOST: str = "qdrant-service"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "omnirag_collection"
    
    # Graph DB (Neo4j)
    NEO4J_URI: str = "bolt://neo4j-cluster:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str # Sensitive
    
    # AWS S3 (Documents) / MinIO (local S3-API-compatible substitute)
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str
    # None = real AWS. Set to MinIO's local URL for dev/demo — same boto3
    # client code path either way (libs/utils/s3_client.py).
    S3_ENDPOINT_URL: Optional[str] = None
    # Was in .env.example but never declared here — silently ignored by
    # pydantic-settings' extra="ignore" until s3_client.py needed them.
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None

    # Ray Job Submission API (ingestion cluster — local head in dev/demo,
    # real multi-node cluster at prod scale; same JobSubmissionClient call)
    RAY_ADDRESS: str = "http://localhost:8265"
    
    # Ray Serve (Internal LLM/Embeddings) — frozen until Phase 11
    RAY_LLM_ENDPOINT: str = "http://llm-service:8000/llm"
    RAY_EMBEDDING_ENDPOINT: str = "http://embed-service:8000/embed"

    # --- LLM Backend Selection (Phase 1) ---
    # api = Groq/OpenRouter auto-failover pair (live demo default)
    # ollama | vllm_local | vllm_modal = manual-select only, never auto-triggered
    LLM_BACKEND: str = "api"
    # Only relevant when LLM_BACKEND=api. The other of the pair is automatic
    # backup on a retryable failure (429/5xx/timeout/connection).
    API_PRIMARY: str = "groq"

    # Groq (primary API backend)
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODELS: Annotated[List[str], NoDecode] = []

    # OpenRouter (backup API backend)
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_MODELS: Annotated[List[str], NoDecode] = []

    # Ollama (local Mac dev, offline, $0). One live test only — Phase 9.
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODELS: Annotated[List[str], NoDecode] = []

    # vLLM — manual-select only. Metal = local Mac; Modal = cloud GPU.
    VLLM_METAL_URL: str = "http://localhost:8001"
    VLLM_METAL_MODELS: Annotated[List[str], NoDecode] = []
    VLLM_MODAL_URL: Optional[str] = None
    VLLM_MODAL_API_KEY: Optional[str] = None
    VLLM_MODAL_MODELS: Annotated[List[str], NoDecode] = []

    # Session & corpus isolation (Phase 2). No login — corpus_id is issued
    # via httpOnly cookie; this is the sliding-window inactivity TTL for it.
    SESSION_TTL_MINUTES: int = 60

    # --- Embeddings + Reranker (Phase 3+4) ---
    # Local primary: FastEmbed (ONNX, CPU, $0), same code path for both
    # query-time (FastAPI) and ingestion-time (Ray worker) embedding.
    # bge-m3 (originally locked) isn't in FastEmbed's supported model set
    # (checked against fastembed 0.8.0 — no multi-vector bge-m3 export
    # exists there) — bge-large-en-v1.5 is the closest-quality substitute
    # FastEmbed actually ships. 1024-dim; Qdrant collection sized to match.
    FASTEMBED_MODEL: str = "BAAI/bge-large-en-v1.5"
    FASTEMBED_RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    # Fallback on local failure — OpenRouter free-tier models, reuses
    # OPENROUTER_API_KEY from the LLM backend config above.
    OPENROUTER_EMBED_MODEL: str = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
    OPENROUTER_RERANK_MODEL: str = "nvidia/llama-nemotron-rerank-vl-1b-v2:free"

    # Retrieval routing: corpora with more than this many distinct
    # documents also query Neo4j (multi-hop/citation questions), not just
    # Qdrant — a single-paper corpus has no graph worth querying.
    GRAPH_SEARCH_MIN_DOCUMENTS: int = 2

    # RRF fusion + reranking
    RRF_K: int = 60  # standard RRF damping constant
    RERANK_TOP_N: int = 5  # final chunk count returned to the LLM after rerank

    # --- Caching (Phase 5, Redis Stack) ---
    # L1 exact-match always applies. L2 semantic-match (RediSearch vector
    # KNN) only applies to RAG-sourced answers (vector_search/graph_search)
    # — never sandbox/web_search, per the locked design (staleness/
    # precision risk for those).
    SEMANTIC_CACHE_THRESHOLD: float = 0.85
    CACHE_VERSIONS_KEPT: int = 2  # historical corpus_version entries kept per cache key (before/after comparison)

    # PDF image description (Phase 3) — one vision call per extracted
    # figure via Gemini (free-tier, multimodal), gated so a broken key
    # doesn't fail ingestion. Not part of the Groq/OpenRouter chat pool —
    # separate budget, no backup configured (see gemini_client.py).
    PDF_DESCRIBE_IMAGES: bool = True
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_VISION_MODEL: str = "gemini-3.5-flash-lite"

    # --- Ingestion execution backend ---
    # in_process (default): runs inline in the FastAPI process as a
    # background task, no separate Ray cluster needed for local/demo dev.
    # ray: the original pipelines/ingestion/main.py Ray Data path (frozen,
    # kept for later re-activation — see PROGRESS.md carry-over notes).
    INGESTION_BACKEND: str = "in_process"
    # Chunks per GraphExtractor LLM call — batching multiple chunks into
    # one call is what keeps this from being 1 Groq call per chunk.
    # 2, not the originally-planned 8-10: live testing showed
    # qwen/qwen3.8-27b has a hard 1000-output-token-per-request ceiling
    # (not a volume/throttling limit — ANY single request asking for
    # >=1000 max_tokens is rejected outright, every time). At ~350 tokens/
    # chunk for graph JSON output, batch_size=8 (2800 tokens) made qwen
    # permanently unable to serve batch calls at all. 2 keeps requests
    # comfortably under every configured model's real ceiling.
    GRAPH_EXTRACTION_BATCH_SIZE: int = 2

    # --- OpenRouter fallback toggle (LLM + reranker) ---
    # False (default): Groq/FastEmbed run with no backup. Config-gated,
    # not deleted — a functioning OpenRouter path is still worth keeping
    # for whenever budget/reliability calls for it again.
    ENABLE_OPENROUTER_FALLBACK: bool = False

    # --- Heuristic model-tier routing (responder.py's final-answer call
    # only — planner/query-rewriter/graph-extraction are fixed-shape
    # tasks that don't need tiering). Routing logic itself is fixed in
    # model_router.py; only the models each tier maps to are env-driven.
    ENABLE_MODEL_ROUTING: bool = True
    MODEL_TIER_SIMPLE: str = "openai/gpt-oss-20b"
    MODEL_TIER_COMPLEX: str = "openai/gpt-oss-120b"
    # Combined query+context char count above which a query is routed to
    # the COMPLEX tier (plus a fixed keyword check — see model_router.py).
    MODEL_ROUTING_CHAR_THRESHOLD: int = 1500

    # --- Logging ---
    LOG_DIR: str = "logs"
    LOG_FILE_NAME: str = "api.log"
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10MB per file before rotating
    LOG_BACKUP_COUNT: int = 5              # keep 5 rotated files (50MB total ceiling)

    # --- Ingestion debug artifacts ---
    # Writes intermediate pipeline output (parsed chunks, extracted graph
    # data) to disk so quality can actually be inspected, not just
    # inferred from log line counts. On by default for local dev; turn
    # off before the public demo (writes raw document content to disk).
    INGEST_DEBUG_DUMP: bool = True
    INGEST_DEBUG_DIR: str = "logs/ingest_debug"

    @field_validator(
        "GROQ_MODELS", "OPENROUTER_MODELS", "OLLAMA_MODELS",
        "VLLM_METAL_MODELS", "VLLM_MODAL_MODELS",
        mode="before",
    )
    @classmethod
    def _split_priority_list(cls, v):
        """.env stores these as a single comma-separated, priority-sorted
        string (first = tried first); split into a list for the client code."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    class Config:
        env_file = ".env" if os.path.exists(".env") else None
        extra = "ignore"

# Instantiate singleton
settings = Settings()
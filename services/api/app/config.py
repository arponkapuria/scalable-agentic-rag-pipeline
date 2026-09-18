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

    # Sandbox code-exec service (services/sandbox/). Defaults to
    # localhost with a published port for local Docker Compose dev,
    # matching every other service URL's pattern here (DATABASE_URL,
    # REDIS_URL, etc. are all .env-configurable, never hardcoded).
    # Real bug found live: sandbox.py previously hardcoded
    # "http://sandbox-service:8080" — a K8s-internal-DNS-style hostname
    # left over from the original cloud-scale design, never adapted when
    # local dev (host-run API + Docker Compose) was introduced. It could
    # never resolve from the host-run `make dev` process, so every
    # sandbox call failed with a connection error — which the responder's
    # conversational LLM then glossed as "I don't have the ability to run
    # code," misrepresenting an infra gap as a capability limitation.
    # Override to the K8s Service DNS name for a real cluster deployment.
    SANDBOX_URL: str = "http://localhost:8080/execute"
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
    # Explicit, persistent cache dir for ALL FastEmbed models (dense
    # embedder, sparse/BM25 embedder, reranker) — FastEmbed's own default
    # (cache_dir=None) falls back to a subdirectory of the SYSTEM TEMP
    # dir (documented upstream issue: qdrant/fastembed#569). On macOS,
    # /var/folders/*/T/ gets cleaned by the OS — this is exactly what
    # produced the "NO_SUCHFILE ... model.onnx failed" crash mid-request:
    # the reranker's cached weights were wiped out from under a running
    # app between sessions, not a corrupted download. `~/.cache/fastembed`
    # survives reboots/cleanup like `~/.cache/huggingface` (Docling's own
    # cache location) already does.
    FASTEMBED_CACHE_DIR: str = os.path.expanduser("~/.cache/fastembed")
    # Fallback on local failure — OpenRouter free-tier models, reuses
    # OPENROUTER_API_KEY from the LLM backend config above.
    OPENROUTER_EMBED_MODEL: str = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
    OPENROUTER_RERANK_MODEL: str = "nvidia/llama-nemotron-rerank-vl-1b-v2:free"

    # RRF fusion + reranking
    RRF_K: int = 60  # standard RRF damping constant
    RERANK_TOP_N: int = 5  # final chunk count returned to the LLM after rerank

    # --- Caching (Phase 5, Redis Stack) ---
    # L1 exact-match always applies. L2 semantic-match (RediSearch vector
    # KNN) only applies to RAG-sourced answers (vector_search) — never
    # sandbox/web_search, per the locked design (staleness/precision risk
    # for those).
    SEMANTIC_CACHE_THRESHOLD: float = 0.85
    # Secondary gate alongside cosine similarity (Phase 6 follow-up) — see
    # redis_cache.py::get_semantic's docstring. 0.5 chosen against a real
    # observed false positive: "different from" vs. "improve upon" (same
    # two entities, different question) scored ~0.27 Jaccard overlap,
    # comfortably below this bar, while genuine paraphrases of the same
    # question typically share most content words. Conservative by
    # design — a miss just costs a full pipeline run, not a wrong answer.
    SEMANTIC_CACHE_LEXICAL_OVERLAP: float = 0.5
    # Second, separate gate from lexical overlap above — catches a
    # DIFFERENT failure mode. Jaccard overlap alone is structurally blind
    # to containment: if Q2's tokens are a strict subset of cached-Q1's
    # tokens (e.g. Q1="A and B", Q2="A" alone), overlap = |Q2|/|Q1| is
    # high by construction regardless of whether Q2 is a genuine
    # paraphrase or a narrower sub-question of a compound one. Live
    # cases: "how is it different from lstm?" (7 tokens) against a cached
    # "...different from lstm and how does it improve from lstm?" (10
    # tokens, Q2's 7 fully contained) scored 0.70 overlap; "does it
    # improve lstm?" (5 tokens) scored 0.50 — both incorrectly served the
    # full compound answer for a narrower question. A true paraphrase of
    # the same question is usually similar in content-word count (same
    # information, different wording); a sub-clause of a compound
    # question is not (it conveys less) — length ratio targets exactly
    # that distinction, which overlap cannot.
    SEMANTIC_CACHE_MIN_LENGTH_RATIO: float = 0.8
    CACHE_VERSIONS_KEPT: int = 2  # historical corpus_version entries kept per cache key (before/after comparison)
    # Second, independent invalidation axis from corpus_version — tracks
    # CODE/logic changes (e.g. a responder bug fix), not data changes.
    # Baked directly into the cache key (Phase 6) so bumping it instantly
    # orphans every previously-cached entry, L1 and L2 both. Manual for
    # now (bump by hand on a behavior-affecting deploy) — deferred, more
    # correct version is deriving this from the git commit SHA at
    # build/startup so it self-invalidates every deploy automatically;
    # not implemented yet (no CI/CD hook in this project).
    CACHE_SCHEMA_VERSION: str = "1"
    # Backstop, not the primary invalidation mechanism — corpus_version/
    # CACHE_SCHEMA_VERSION/cascade-delete are. This just bounds the worst
    # case if any of those ever miss an entry. Generous on purpose (24h
    # default) so it never expires a legitimately-fresh entry mid-demo.
    CACHE_TTL_SECONDS: int = 86400

    # PDF image description (Phase 3) — one vision call per extracted
    # figure via Gemini (free-tier, multimodal), gated so a broken key
    # doesn't fail ingestion. Not part of the Groq/OpenRouter chat pool —
    # separate budget, no backup configured (see gemini_client.py).
    PDF_DESCRIBE_IMAGES: bool = True
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_VISION_MODEL: str = "gemini-3.5-flash-lite"

    # --- Docling parsing/chunking (in-process pipeline only — see
    # loaders/docling_loader.py) ---
    # Off by default: literature-survey PDFs are assumed born-digital.
    # RapidOCR (Docling's default OCR engine) uses ONNXRuntime, not torch,
    # so flipping this on doesn't add a torch dependency — safe to enable
    # if you hit scanned documents.
    PDF_ENABLE_OCR: bool = False
    # TableFormer has no ONNX path in mainline Docling (confirmed via
    # Docling's own technical report — PyTorch-only for inference), so
    # this only trades speed for quality, not resource type. "accurate"
    # matches the project's accuracy-first priority ordering; "fast" is
    # there for faster local iteration if table-heavy PDFs are slow on
    # the M1.
    TABLE_STRUCTURE_MODE: str = "accurate"  # "accurate" | "fast"
    # Derived from FASTEMBED_MODEL's own tokenizer if unset — HybridChunker
    # needs a token ceiling that actually matches the embedder, not an
    # independent guess.
    CHUNK_MAX_TOKENS: int = 512
    # Comma-separated picture-classifier labels to skip captioning for
    # (saves a Gemini call per skipped image). Empty by default — the
    # classifier's real label set isn't confirmed against a live run yet;
    # check logs/ingest_debug/{corpus_id}/docling_pictures.json after a
    # real ingestion and populate this once you've seen actual labels
    # (e.g. "logo,icon") rather than guessing.
    PICTURE_SKIP_CLASSES: str = ""

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

    @field_validator("FASTEMBED_CACHE_DIR", mode="before")
    @classmethod
    def _expand_cache_dir(cls, v):
        """~ isn't shell-expanded when read from a .env file (python-dotenv
        treats it as a literal character) — expanduser here guarantees a
        real path regardless of whether this came from the Python default
        above or an actual .env override."""
        return os.path.expanduser(v) if isinstance(v, str) else v

    class Config:
        env_file = ".env" if os.path.exists(".env") else None
        extra = "ignore"

# Instantiate singleton
settings = Settings()
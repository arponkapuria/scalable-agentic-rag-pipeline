"""
Application configuration, loaded from environment variables / .env. Every tunable used anywhere in the app — LLM backend selection, embeddings, retrieval, caching, ingestion, logging, and evaluation — lives in this one settings object.
"""
import os
from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from typing import Annotated, List, Optional


class Settings(BaseSettings):
    """Application configuration. Reads environment variables automatically (case-insensitive)."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ENV: str = "prod"
    LOG_LEVEL: str = "INFO"

    DATABASE_URL: str

    REDIS_URL: str

    QDRANT_HOST: str = "qdrant-service"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "omnirag_collection"

    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str
    S3_ENDPOINT_URL: Optional[str] = None  # None = real AWS; set for MinIO
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None

    # api = Groq/OpenRouter auto-failover pair. ollama/vllm_local/vllm_modal = manual-select only.
    LLM_BACKEND: str = "api"
    API_PRIMARY: str = "groq"  # only relevant when LLM_BACKEND=api

    GROQ_API_KEY: Optional[str] = None
    GROQ_MODELS: Annotated[List[str], NoDecode] = []

    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_MODELS: Annotated[List[str], NoDecode] = []

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODELS: Annotated[List[str], NoDecode] = []

    VLLM_METAL_URL: str = "http://localhost:8001"
    VLLM_METAL_MODELS: Annotated[List[str], NoDecode] = []
    VLLM_MODAL_URL: Optional[str] = None
    VLLM_MODAL_API_KEY: Optional[str] = None
    VLLM_MODAL_MODELS: Annotated[List[str], NoDecode] = []

    SANDBOX_URL: str = "http://localhost:8080/execute"

    SESSION_TTL_MINUTES: int = 60

    # FastEmbed (ONNX, CPU) is the sole embedder.
    FASTEMBED_MODEL: str = "BAAI/bge-large-en-v1.5"
    FASTEMBED_RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    FASTEMBED_CACHE_DIR: str = os.path.expanduser("~/.cache/fastembed")  # explicit, not the system temp dir
    OPENROUTER_EMBED_MODEL: str = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
    OPENROUTER_RERANK_MODEL: str = "nvidia/llama-nemotron-rerank-vl-1b-v2:free"

    RRF_K: int = 60
    RERANK_TOP_N: int = 5

    # L1 exact-match always applies. L2 semantic-match only applies to vector_search answers.
    SEMANTIC_CACHE_THRESHOLD: float = 0.85
    SEMANTIC_CACHE_LEXICAL_OVERLAP: float = 0.5
    SEMANTIC_CACHE_MIN_LENGTH_RATIO: float = 0.8
    CACHE_VERSIONS_KEPT: int = 2
    CACHE_SCHEMA_VERSION: str = "1"  # bump to invalidate every entry, independent of corpus_version
    CACHE_TTL_SECONDS: int = 86400

    PDF_DESCRIBE_IMAGES: bool = True
    MISTRAL_API_KEY: Optional[str] = None
    MISTRAL_VISION_MODEL: str = "ministral-14b-2512"
    CAPTION_MIN_INTERVAL_SECONDS: float = 2.0

    PDF_ENABLE_OCR: bool = False
    TABLE_STRUCTURE_MODE: str = "accurate"  # "accurate" | "fast"
    CHUNK_MAX_TOKENS: int = 512
    PICTURE_SKIP_CLASSES: str = ""  # comma-separated classifier labels to skip captioning for

    ENABLE_OPENROUTER_FALLBACK: bool = False

    ENABLE_MODEL_ROUTING: bool = True
    MODEL_TIER_SIMPLE: str = "openai/gpt-oss-20b"
    MODEL_TIER_COMPLEX: str = "openai/gpt-oss-120b"
    MODEL_ROUTING_CHAR_THRESHOLD: int = 1500

    ANSWER_MAX_TOKENS: int = 2048

    LOG_DIR: str = "logs"
    LOG_FILE_NAME: str = "api.log"
    LOG_MAX_BYTES: int = 10 * 1024 * 1024
    LOG_BACKUP_COUNT: int = 5

    INGEST_DEBUG_DUMP: bool = True
    INGEST_DEBUG_DIR: str = "logs/ingest_debug"

    # Evaluation judges — deliberately different vendors from the generator and captioner.
    GOOGLE_API_KEY: Optional[str] = None
    GEMMA_JUDGE_MODEL: str = "gemma-4-31b-it"
    GEMMA_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

    COHERE_API_KEY: Optional[str] = None
    COHERE_JUDGE_MODEL: str = "command-r7b-12-2024"
    COHERE_BASE_URL: str = "https://api.cohere.ai/compatibility/v1"

    @field_validator(
        "GROQ_MODELS", "OPENROUTER_MODELS", "OLLAMA_MODELS",
        "VLLM_METAL_MODELS", "VLLM_MODAL_MODELS",
        mode="before",
    )
    @classmethod
    def _split_priority_list(cls, v):
        """Splits a comma-separated, priority-ordered model list from .env into a list.

        Args:
            v: Comma-separated string, or already a list.

        Returns:
            A list of model ids, first = tried first.
        """
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @field_validator("FASTEMBED_CACHE_DIR", mode="before")
    @classmethod
    def _expand_cache_dir(cls, v):
        """Expands a leading ~, since .env files don't shell-expand it.

        Args:
            v: The raw path value.

        Returns:
            The expanded absolute path.
        """
        return os.path.expanduser(v) if isinstance(v, str) else v


settings = Settings()
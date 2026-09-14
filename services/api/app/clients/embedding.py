"""
Async embedding client used by query-time code (chat retrieval, semantic
cache) and, via the same interface, by ingestion.

FastEmbed (bge-large-en-v1.5, 1024-dim) is the SOLE embedder — no
OpenRouter fallback. Two reasons, both non-negotiable given accuracy is
the top priority here:

  1. Dimension safety: Qdrant's collection is fixed at 1024-dim.
     OpenRouter's nemotron-embed-vl returns 2048-dim vectors — a fallback
     that "succeeds" there produces vectors Qdrant rejects at write time
     (ingestion) or search time (retrieval), or in the cache's case,
     silently skips the L2 write. That's not resilience, it's a
     different, more confusing failure mode. A fallback that can corrupt
     or silently degrade retrieval accuracy is worse than no fallback.
  2. Speed/cost: FastEmbed is also genuinely faster here — no network
     round-trip, and it doesn't compete with anything else for the
     Groq/OpenRouter free-tier rate-limit budget during ingestion.

OpenRouterEmbeddingClient is kept below (unused, not wired into
embedding_client) as ready-made groundwork IF a same-dimension API
provider is ever swapped in — e.g. Cohere embed-english-v3.0 or Jina
embeddings-v3, both natively 1024-dim, unlike nemotron-embed-vl. Swapping
requires changing its base_url/model/response-parsing to that provider's
API shape, then re-adding it as backup in the FailoverEmbeddingClient
below.

Retry-with-backoff, empty-output detection, and circuit breaker are still
implemented on both classes below (kept from the fallback-parity pattern
built for the LLM client) — genuinely useful if FastEmbed itself needs
retry semantics, independent of whether a fallback exists.
"""
import asyncio
import logging
from abc import ABC, abstractmethod

import httpx
from models.embeddings.fastembed_client import fastembed_client
from services.api.app.config import settings
from libs.utils.backoff import exponential_backoff
from libs.utils.circuit_breaker import CircuitBreaker, CircuitOpenError

logger = logging.getLogger(__name__)


class EmbeddingClient(ABC):
    @abstractmethod
    async def embed_query(self, text: str) -> list[float]: ...

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedQueryClient(EmbeddingClient):
    """Wraps the sync FastEmbed model with asyncio.to_thread for FastAPI request handlers."""

    async def embed_query(self, text: str) -> list[float]:
        vectors = await asyncio.to_thread(fastembed_client.embed_dense, [text])
        return vectors[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = await asyncio.to_thread(fastembed_client.embed_dense, texts)
        if not vectors or any(not v for v in vectors):
            raise ValueError("FastEmbed returned empty vector(s)")
        return vectors


class OpenRouterEmbeddingClient(EmbeddingClient):
    """
    NOT currently used (see module docstring) — nemotron-embed-vl's
    2048-dim output is incompatible with Qdrant's fixed 1024-dim
    collection. Kept as a template for swapping in a real same-dimension
    provider (Cohere/Jina) later, not as an active fallback.
    """
    def __init__(self):
        self._client: httpx.AsyncClient | None = None
        self._circuit = CircuitBreaker(failure_threshold=3, cooldown_seconds=30.0)

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url="https://openrouter.ai/api/v1",
                headers={"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}"},
                timeout=30.0,
            )
        return self._client

    @exponential_backoff(max_retries=2)
    async def _post(self, inputs: list[str]) -> httpx.Response:
        response = await self._get_client().post(
            "/embeddings",
            json={"model": settings.OPENROUTER_EMBED_MODEL, "input": inputs},
        )
        response.raise_for_status()
        return response

    async def _embed(self, inputs: list[str]) -> list[list[float]]:
        self._circuit.before_call()  # raises CircuitOpenError if cooling down
        # Defensive cast: a numpy array (e.g. from an unconverted Ray Data
        # batch) is a plausible caller mistake and json.dumps can't
        # serialize it at all — cheap insurance against a repeat of that.
        inputs = [str(x) for x in inputs]
        try:
            response = await self._post(inputs)
            data = response.json()["data"]
            vectors = [row["embedding"] for row in data]
            if not vectors or any(not v for v in vectors):
                raise ValueError("OpenRouter returned empty embedding(s)")
            self._circuit.record_success()
            return vectors
        except Exception:
            self._circuit.record_failure()
            raise

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text]))[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts)


class FailoverEmbeddingClient(EmbeddingClient):
    """
    NOT currently used (see module docstring) — pairs with
    OpenRouterEmbeddingClient above; both are reactivated together if a
    same-dimension provider ever replaces it as backup.
    """
    def __init__(self, primary: EmbeddingClient, backup: EmbeddingClient):
        self.primary = primary
        self.backup = backup

    async def embed_query(self, text: str) -> list[float]:
        try:
            return await self.primary.embed_query(text)
        except (Exception, CircuitOpenError) as e:
            logger.warning(f"Primary embedder failed, falling back: {e}")
            return await self.backup.embed_query(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return await self.primary.embed_documents(texts)
        except (Exception, CircuitOpenError) as e:
            logger.warning(f"Primary embedder failed, falling back: {e}")
            return await self.backup.embed_documents(texts)


# FastEmbed only — no fallback (see module docstring for why OpenRouter
# can't safely serve that role given the dimension constraint).
embedding_client: EmbeddingClient = FastEmbedQueryClient()
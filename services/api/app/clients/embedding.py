"""
Async embedding client. FastEmbed is the sole active embedder — no fallback is wired in, since Qdrant's collection is fixed at one vector dimension and a mismatched fallback would corrupt retrieval. An OpenRouter-based fallback is kept below as an unused template.
"""
import asyncio
import logging
from abc import ABC, abstractmethod

import httpx
from services.api.app.clients.fastembed_client import fastembed_client
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
    """Wraps the synchronous FastEmbed model with asyncio.to_thread for async callers."""

    async def embed_query(self, text: str) -> list[float]:
        """Embeds a single query string.

        Args:
            text: The text to embed.

        Returns:
            The dense embedding vector.
        """
        vectors = await asyncio.to_thread(fastembed_client.embed_dense, [text])
        return vectors[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds a batch of document texts.

        Args:
            texts: The texts to embed.

        Returns:
            One dense embedding vector per input text.

        Raises:
            ValueError: If any returned vector is empty.
        """
        vectors = await asyncio.to_thread(fastembed_client.embed_dense, texts)
        if not vectors or any(not v for v in vectors):
            raise ValueError("FastEmbed returned empty vector(s)")
        return vectors


class OpenRouterEmbeddingClient(EmbeddingClient):
    """Not currently used — template for a future same-dimension embedding provider."""

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
        self._circuit.before_call()
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
    """Not currently used — pairs with OpenRouterEmbeddingClient."""

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


embedding_client: EmbeddingClient = FastEmbedQueryClient()
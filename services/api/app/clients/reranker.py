"""
Async reranker client. FastEmbed's cross-encoder is primary; an optional OpenRouter prompted-LLM reranker is a config-gated fallback.
"""
import asyncio
import json
import logging
from abc import ABC, abstractmethod

import httpx
from services.api.app.clients.fastembed_reranker import fastembed_reranker
from services.api.app.config import settings
from libs.utils.backoff import exponential_backoff
from libs.utils.circuit_breaker import CircuitBreaker, CircuitOpenError

logger = logging.getLogger(__name__)

RERANK_PROMPT = """Score how relevant each document is to the query, from 0.0 (irrelevant) to 1.0 (highly relevant).

Query: {query}

Documents:
{documents}

Output JSON only: {{"scores": [<float>, ...]}} in the same order as the documents above."""


class RerankerClient(ABC):
    @abstractmethod
    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Returns relevance scores in the same order as `documents`."""
        ...


class FastEmbedRerankClient(RerankerClient):
    """Reranks using FastEmbed's local cross-encoder model."""

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Scores each document's relevance to the query.

        Args:
            query: The search query.
            documents: Candidate document texts.

        Returns:
            One relevance score per document, same order as input.

        Raises:
            ValueError: If the scores returned don't match the document count.
        """
        scores = await asyncio.to_thread(fastembed_reranker.rerank, query, documents)
        if not scores or len(scores) != len(documents):
            raise ValueError("FastEmbed reranker returned malformed scores")
        return scores


class OpenRouterRerankClient(RerankerClient):
    """Reranks using a prompted LLM call scoring all documents in one response."""

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
    async def _post(self, query: str, documents: list[str]) -> httpx.Response:
        doc_list = "\n".join(f"[{i}] {doc[:500]}" for i, doc in enumerate(documents))
        response = await self._get_client().post(
            "/chat/completions",
            json={
                "model": settings.OPENROUTER_RERANK_MODEL,
                "messages": [{"role": "user", "content": RERANK_PROMPT.format(query=query, documents=doc_list)}],
                "temperature": 0.0,
            },
        )
        response.raise_for_status()
        return response

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Scores each document's relevance via a prompted LLM call.

        Args:
            query: The search query.
            documents: Candidate document texts.

        Returns:
            One relevance score per document, same order as input.

        Raises:
            ValueError: If the response is empty or the score count doesn't match.
        """
        self._circuit.before_call()
        try:
            response = await self._post(query, documents)
            content = response.json()["choices"][0]["message"]["content"]
            if not content or not content.strip():
                raise ValueError("OpenRouter reranker returned empty content")
            scores = json.loads(content)["scores"]
            if len(scores) != len(documents):
                raise ValueError(f"Reranker returned {len(scores)} scores for {len(documents)} documents")
            self._circuit.record_success()
            return scores
        except Exception:
            self._circuit.record_failure()
            raise


class FailoverRerankerClient(RerankerClient):
    """Wraps a primary and backup reranker, falling back on any primary failure."""

    def __init__(self, primary: RerankerClient, backup: RerankerClient):
        self.primary = primary
        self.backup = backup

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        try:
            return await self.primary.rerank(query, documents)
        except (Exception, CircuitOpenError) as e:
            logger.warning(f"Primary reranker failed, falling back: {e}")
            return await self.backup.rerank(query, documents)


reranker_client: RerankerClient = (
    FailoverRerankerClient(primary=FastEmbedRerankClient(), backup=OpenRouterRerankClient())
    if settings.ENABLE_OPENROUTER_FALLBACK
    else FastEmbedRerankClient()
)
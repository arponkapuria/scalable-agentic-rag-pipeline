"""
Async reranker client, same primary/backup shape and fallback-parity
pattern as embedding.py (retry-with-backoff, empty-output detection,
circuit breaker). FastEmbed (bge-reranker-base) is PRIMARY, OpenRouter is
the fallback.

Why FastEmbed-first (accuracy priority): bge-reranker-base is a genuine
cross-encoder — query and document are scored together in one model pass,
the standard/correct reranking architecture. The OpenRouter path below is
NOT a true cross-encoder call: OpenRouter's public docs don't document a
generic /rerank endpoint for this model, so it's implemented as a
prompted LLM scoring call (ask the model to output JSON relevance scores
via /chat/completions) — measurably less reliable than a real
cross-encoder (score drift, inconsistent calibration, JSON malformation
under load), so it's the fallback, not primary. Also faster/free like
embedding.py's equivalent choice — no network round-trip, no competing
for the same Groq/OpenRouter rate-limit budget GraphExtractor stresses.
"""
import asyncio
import json
import logging
from abc import ABC, abstractmethod

import httpx
from models.rerankers.fastembed_reranker import fastembed_reranker
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
    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        scores = await asyncio.to_thread(fastembed_reranker.rerank, query, documents)
        if not scores or len(scores) != len(documents):
            raise ValueError("FastEmbed reranker returned malformed scores")
        return scores


class OpenRouterRerankClient(RerankerClient):
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
        self._circuit.before_call()  # raises CircuitOpenError if cooling down
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
    def __init__(self, primary: RerankerClient, backup: RerankerClient):
        self.primary = primary
        self.backup = backup

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        try:
            return await self.primary.rerank(query, documents)
        except (Exception, CircuitOpenError) as e:
            logger.warning(f"Primary reranker failed, falling back: {e}")
            return await self.backup.rerank(query, documents)


# FastEmbed primary (real cross-encoder). OpenRouter fallback is
# config-gated off by default (ENABLE_OPENROUTER_FALLBACK) — see
# PROGRESS.md carry-over notes; kept, not deleted, for when it's needed again.
reranker_client: RerankerClient = (
    FailoverRerankerClient(primary=FastEmbedRerankClient(), backup=OpenRouterRerankClient())
    if settings.ENABLE_OPENROUTER_FALLBACK
    else FastEmbedRerankClient()
)
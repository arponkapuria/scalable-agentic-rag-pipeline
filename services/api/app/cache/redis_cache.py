"""
Two-layer response cache backed by Redis Stack. L1 is an exact-match Redis GET keyed by corpus_id, schema version, and a query hash. L2 is a RediSearch vector KNN semantic match, gated by cosine similarity plus lexical overlap and length-ratio checks, used only for vector_search answers.
"""
import hashlib
import json
import logging
import re
import struct
from datetime import datetime, timezone
from typing import Any, Optional

import redis.asyncio as redis
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from services.api.app.config import settings
from services.api.app.clients.embedding import embedding_client

logger = logging.getLogger(__name__)

VECTOR_DIM = 1024
INDEX_NAME = "idx:semantic_cache"
KEY_PREFIX = "cache:"

_TAG_ESCAPE_RE = re.compile(r'([,.<>{}\[\]"\':;!@#$%^&*()\-+=~ ])')


def _escape_tag_value(value: str) -> str:
    """Escapes RediSearch's special characters for use inside a TAG {} filter.

    Args:
        value: The raw value to escape.

    Returns:
        The escaped value.
    """
    return _TAG_ESCAPE_RE.sub(r"\\\1", value)


class RedisCache:
    """Reads and writes the two-layer (exact + semantic) response cache."""

    def __init__(self):
        self._redis: Optional[redis.Redis] = None
        self._index_ready = False

    def _get_client(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.from_url(settings.REDIS_URL, decode_responses=False)
        return self._redis

    async def _ensure_index(self):
        """Creates the RediSearch vector index on first use. Idempotent."""
        if self._index_ready:
            return
        client = self._get_client()
        try:
            await client.ft(INDEX_NAME).info()
        except Exception:
            await client.ft(INDEX_NAME).create_index(
                fields=[
                    TagField("corpus_id"),
                    VectorField(
                        "vector",
                        "HNSW",
                        {"TYPE": "FLOAT32", "DIM": VECTOR_DIM, "DISTANCE_METRIC": "COSINE"},
                    ),
                ],
                definition=IndexDefinition(prefix=[KEY_PREFIX], index_type=IndexType.HASH),
            )
        self._index_ready = True

    @staticmethod
    def _normalize(query: str) -> str:
        return " ".join(query.strip().lower().split())

    @classmethod
    def _l1_key(cls, corpus_id: str, query: str) -> str:
        """Builds the L1 cache key: cache:{corpus_id}:{schema_version}:{hash(query)}.

        Args:
            corpus_id: The session's corpus id.
            query: The raw user query.

        Returns:
            The full Redis key.
        """
        digest = hashlib.sha256(cls._normalize(query).encode()).hexdigest()
        return f"{KEY_PREFIX}{corpus_id}:{settings.CACHE_SCHEMA_VERSION}:{digest}"

    async def _get_corpus_version(self, corpus_id: str) -> int:
        client = self._get_client()
        raw = await client.get(f"corpus_version:{corpus_id}".encode())
        return int(raw) if raw else 0

    async def get_exact(self, corpus_id: str, query: str) -> dict[str, Any] | None:
        """Looks up the entry at the current corpus_version.

        Args:
            corpus_id: The session's corpus id.
            query: The raw user query.

        Returns:
            The matching entry dict, or None.
        """
        entries = await self._get_entries(corpus_id, query)
        if not entries:
            return None
        current_version = await self._get_corpus_version(corpus_id)
        for entry in entries:
            if entry["corpus_version"] == current_version:
                return entry
        return None

    async def get_all_versions(self, corpus_id: str, query: str) -> list[dict[str, Any]]:
        """Returns every cached entry for this query, across versions, newest first.

        Args:
            corpus_id: The session's corpus id.
            query: The raw user query.

        Returns:
            List of entry dicts, possibly empty.
        """
        return await self._get_entries(corpus_id, query)

    async def _get_entries(self, corpus_id: str, query: str) -> list[dict[str, Any]]:
        try:
            client = self._get_client()
            raw = await client.hget(self._l1_key(corpus_id, query), b"entries")
            if not raw:
                return []
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"L1 cache lookup failed: {e}")
            return []

    async def set_exact(
        self,
        corpus_id: str,
        query: str,
        answer: str,
        sources: list[str],
        tool_used: str,
        backend_used: str,
        model_used: str = "",
    ):
        """Writes a new cache entry, and the matching L2 vector entry if the tool is cache-eligible.

        Args:
            corpus_id: The session's corpus id.
            query: The raw user query.
            answer: The generated answer text.
            sources: Filenames cited in the answer.
            tool_used: Which tool produced the answer.
            backend_used: Which LLM provider answered.
            model_used: Which specific model answered.
        """
        try:
            client = self._get_client()
            current_version = await self._get_corpus_version(corpus_id)
            entries = await self._get_entries(corpus_id, query)
            entries = [e for e in entries if e["corpus_version"] != current_version]
            entries.insert(
                0,
                {
                    "corpus_version": current_version,
                    "answer": answer,
                    "sources": sources,
                    "tool_used": tool_used,
                    "backend_used": backend_used,
                    "model_used": model_used,
                    "cached_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            entries = entries[: settings.CACHE_VERSIONS_KEPT]

            key = self._l1_key(corpus_id, query)
            mapping = {
                b"entries": json.dumps(entries).encode(),
                b"corpus_id": corpus_id.encode(),
                b"query": self._normalize(query).encode(),
            }

            if tool_used == "vector_search":
                vector = await self._embed_for_cache(query)
                if vector is not None:
                    mapping[b"vector"] = vector

            await client.hset(key, mapping=mapping)
            await client.expire(key, settings.CACHE_TTL_SECONDS)

        except Exception as e:
            logger.warning(f"Failed to write cache entry: {e}")

    async def _embed_for_cache(self, query: str) -> Optional[bytes]:
        """Embeds a query and packs it as raw float32 bytes for RediSearch's VECTOR field.

        Args:
            query: The raw user query.

        Returns:
            Packed vector bytes, or None on failure or a dimension mismatch.
        """
        try:
            vector = await embedding_client.embed_query(query)
            if len(vector) != VECTOR_DIM:
                logger.warning(f"Embedder returned {len(vector)}-dim vector, expected {VECTOR_DIM} — skipping L2 write.")
                return None
            return struct.pack(f"{VECTOR_DIM}f", *vector)
        except Exception as e:
            logger.warning(f"Failed to embed query for cache: {e}")
            return None

    @staticmethod
    def _lexical_overlap(query_a: str, query_b: str) -> float:
        """Computes Jaccard token overlap between two normalized queries.

        Args:
            query_a: First normalized query.
            query_b: Second normalized query.

        Returns:
            Overlap ratio from 0.0 to 1.0.
        """
        tokens_a, tokens_b = set(query_a.split()), set(query_b.split())
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    @staticmethod
    def _length_ratio(query_a: str, query_b: str) -> float:
        """Computes shorter-to-longer token-count ratio, catching sub-clause containment that lexical overlap alone can't.

        Args:
            query_a: First normalized query.
            query_b: Second normalized query.

        Returns:
            Ratio from 0.0 to 1.0.
        """
        len_a, len_b = len(query_a.split()), len(query_b.split())
        if len_a == 0 or len_b == 0:
            return 0.0
        return min(len_a, len_b) / max(len_a, len_b)

    async def get_semantic(self, corpus_id: str, query: str) -> dict[str, Any] | None:
        """Finds a semantically similar cached answer via vector KNN, gated by similarity + lexical overlap + length ratio.

        Args:
            corpus_id: The session's corpus id.
            query: The raw user query.

        Returns:
            The matching entry dict, or None if no candidate clears every gate.
        """
        await self._ensure_index()
        try:
            query_vector = await embedding_client.embed_query(query)
            if len(query_vector) != VECTOR_DIM:
                return None
            vector_bytes = struct.pack(f"{VECTOR_DIM}f", *query_vector)

            client = self._get_client()
            escaped_corpus_id = _escape_tag_value(corpus_id)
            search_query = (
                Query(f"(@corpus_id:{{{escaped_corpus_id}}})=>[KNN 1 @vector $vec AS score]")
                .sort_by("score")
                .return_fields("score")
                .dialect(2)
            )
            results = await client.ft(INDEX_NAME).search(
                search_query, query_params={"vec": vector_bytes}
            )
            if not results.docs:
                return None

            top = results.docs[0]
            similarity = 1.0 - float(top.score)  # RediSearch COSINE distance -> similarity
            if similarity < settings.SEMANTIC_CACHE_THRESHOLD:
                return None

            matched_key = top.id.encode() if isinstance(top.id, str) else top.id
            raw_entries, raw_query = await client.hmget(matched_key, [b"entries", b"query"])
            if not raw_entries or not raw_query:
                return None  # legacy entry with no stored query text — nothing to compare

            overlap = self._lexical_overlap(self._normalize(query), raw_query.decode())
            if overlap < settings.SEMANTIC_CACHE_LEXICAL_OVERLAP:
                return None

            length_ratio = self._length_ratio(self._normalize(query), raw_query.decode())
            if length_ratio < settings.SEMANTIC_CACHE_MIN_LENGTH_RATIO:
                return None

            entries = json.loads(raw_entries)
            current_version = await self._get_corpus_version(corpus_id)
            for entry in entries:
                if entry["corpus_version"] == current_version:
                    logger.info(f"L2 semantic cache hit (similarity={similarity:.3f}, overlap={overlap:.2f}, length_ratio={length_ratio:.2f})")
                    return entry
            return None

        except Exception as e:
            logger.warning(f"L2 semantic cache lookup failed: {e}")
            return None

    async def delete_by_corpus_id(self, corpus_id: str) -> int:
        """Deletes every cache entry for a corpus_id, via SCAN (non-blocking, safe on a live instance).

        Args:
            corpus_id: The session's corpus id.

        Returns:
            Number of keys deleted.
        """
        client = self._get_client()
        pattern = f"{KEY_PREFIX}{corpus_id}:*".encode()
        deleted = 0
        cursor = 0
        while True:
            cursor, keys = await client.scan(cursor=cursor, match=pattern, count=100)
            if keys:
                deleted += await client.delete(*keys)
            if cursor == 0:
                break
        return deleted

    async def close(self):
        """Closes the Redis connection, if open."""
        if self._redis:
            await self._redis.close()


redis_cache = RedisCache()
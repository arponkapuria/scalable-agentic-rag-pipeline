"""
Phase 5 caching — Redis Stack, not Qdrant (locked design: avoids a second
network hop per lookup, ~1-5ms vs ~10-30ms). Replaces cache/semantic.py,
which pointed at the dead Ray Serve embed endpoint and only did a single
Qdrant-based semantic match with no versioning at all.

Two layers:
  L1 (exact match): plain Redis GET on
    cache:{corpus_id}:{CACHE_SCHEMA_VERSION}:{hash(normalized_query)} —
    corpus_id and CACHE_SCHEMA_VERSION are plaintext key segments (Phase
    6), not hashed in, specifically so cascade-delete can SCAN the key
    space directly by corpus_id and so bumping CACHE_SCHEMA_VERSION
    orphans every old entry by construction. Applies to every answer
    regardless of tool_used.
  L2 (semantic match): RediSearch vector KNN over a corpus_id-tagged
    index, ~0.85 cosine threshold, PLUS a lexical token-overlap gate
    (SEMANTIC_CACHE_LEXICAL_OVERLAP, Phase 6 follow-up) — see get_semantic's
    docstring for why dense similarity alone isn't sufficient. ONLY for
    tool_used == vector_search — never sandbox (numeric precision risk) or
    web_search (staleness risk), per the locked design.

Value schema (one Redis key per L1 hash, value = JSON list, capped at
CACHE_VERSIONS_KEPT entries, newest first):
    [{"corpus_version": int, "answer": str, "sources": [str],
      "tool_used": str, "backend_used": str, "cached_at": iso8601}, ...]

corpus_version comes from pipelines/ingestion/pipeline.py's Redis INCR at
the end of a successful ingestion job — NOT at upload start (locked design:
"the corpus hasn't actually changed until indexing finishes").

On a lookup that finds entries but none at the CURRENT corpus_version, the
caller (chat.py) is expected to run the pipeline fresh and can present
the returned stale entry as a "your earlier answer (before you added X)"
comparison — this module only stores/retrieves, chat.py owns the
before/after UI behavior.

NOT live-verified against a running Redis Stack instance in this
environment — RediSearch's FT.CREATE/KNN query syntax is standard/
well-documented, but the exact vector dimension assumption below (1024,
matching FastEmbed's bge-large-en-v1.5) needs confirming if OpenRouter's
embed model (now primary, see clients/embedding.py) returns a different
dimension — see _embed_for_cache()'s try/except for how that's handled.
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

VECTOR_DIM = 1024  # matches FastEmbed's bge-large-en-v1.5 — see module docstring
INDEX_NAME = "idx:semantic_cache"
KEY_PREFIX = "cache:"

# Characters RediSearch's query parser treats as special even inside a
# TAG {} block (its own documented escape list) — a bare corpus_id UUID
# (e.g. "a7f33ab5-8c0c-4ba1-...") breaks the parser on every hyphen
# without this. Each match gets backslash-escaped before being spliced
# into a query string.
_TAG_ESCAPE_RE = re.compile(r'([,.<>{}\[\]"\':;!@#$%^&*()\-+=~ ])')


def _escape_tag_value(value: str) -> str:
    return _TAG_ESCAPE_RE.sub(r"\\\1", value)


class RedisCache:
    def __init__(self):
        # decode_responses=False here (unlike cache/redis.py's session-
        # storage client) — vector fields are raw float32 bytes, and
        # decode_responses=True would try (and fail) to UTF-8-decode them.
        self._redis: Optional[redis.Redis] = None
        self._index_ready = False

    def _get_client(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.from_url(settings.REDIS_URL, decode_responses=False)
        return self._redis

    async def _ensure_index(self):
        """Idempotent — same lazy-init pattern as qdrant_client.init_collections()."""
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
        """cache:{corpus_id}:{schema_version}:{hash(query)} — corpus_id and
        CACHE_SCHEMA_VERSION are kept as plaintext key segments (Phase 6),
        not hashed in, specifically so cascade-delete can SCAN MATCH
        cache:{corpus_id}:* directly against the key space, without
        depending on the RediSearch index (which exists for L2 semantic
        *search*, a different job, and only covers L2-eligible entries
        anyway). Only the query itself is unbounded-length/arbitrary-
        character, so only it still needs hashing.
        Bumping CACHE_SCHEMA_VERSION changes this key for every future
        write, orphaning every entry under the old version automatically —
        this is what actually makes that field do what the design always
        intended (previously declared but never implemented)."""
        digest = hashlib.sha256(cls._normalize(query).encode()).hexdigest()
        return f"{KEY_PREFIX}{corpus_id}:{settings.CACHE_SCHEMA_VERSION}:{digest}"

    async def _get_corpus_version(self, corpus_id: str) -> int:
        client = self._get_client()
        raw = await client.get(f"corpus_version:{corpus_id}".encode())
        return int(raw) if raw else 0

    # --- L1: exact match ---

    async def get_exact(self, corpus_id: str, query: str) -> dict[str, Any] | None:
        """
        Returns the entry at the CURRENT corpus_version if one exists.
        Callers that also want the most recent STALE entry (for the
        before/after view) should call get_all_versions() instead.
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
        """Newest-first list of cached entries for this exact query, across versions."""
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
        """
        Writes/prepends the new entry, capped at CACHE_VERSIONS_KEPT
        (locked design: no invalidation on upload — old entries are kept,
        not deleted, so the before/after comparison has something to show).
        Also writes the L2 vector entry when tool_used is cache-eligible
        for semantic match (vector_search only).
        """
        try:
            client = self._get_client()
            current_version = await self._get_corpus_version(corpus_id)
            entries = await self._get_entries(corpus_id, query)
            # Drop any existing entry at this exact version (re-caching a
            # repeat question at the same corpus state replaces, not
            # duplicates) then prepend the new one, newest first.
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
                # Normalized query text — Phase 6 addition, needed by
                # get_semantic's lexical-overlap gate (see its docstring).
                # Not used for L1 lookup (the key hash already IS this
                # value) — only read back on an L2 KNN match.
                b"query": self._normalize(query).encode(),
            }

            l2_eligible = tool_used == "vector_search"
            if l2_eligible:
                vector = await self._embed_for_cache(query)
                if vector is not None:
                    mapping[b"vector"] = vector

            await client.hset(key, mapping=mapping)
            # Backstop TTL (Phase 6) — not the primary invalidation
            # mechanism (corpus_version/CACHE_SCHEMA_VERSION/cascade-
            # delete are); this only bounds the worst case if one of
            # those ever misses an entry. Re-set on every write, so an
            # actively-used entry's expiry keeps sliding forward.
            await client.expire(key, settings.CACHE_TTL_SECONDS)

        except Exception as e:
            logger.warning(f"Failed to write cache entry: {e}")

    async def _embed_for_cache(self, query: str) -> Optional[bytes]:
        """Packs the query embedding as raw float32 bytes for RediSearch's
        VECTOR field. Returns None (soft-fail, L1 write still proceeds) if
        the active embedder's output dimension doesn't match VECTOR_DIM —
        see module docstring's flag on OpenRouter vs FastEmbed dimension."""
        try:
            vector = await embedding_client.embed_query(query)
            if len(vector) != VECTOR_DIM:
                logger.warning(
                    f"Embedder returned {len(vector)}-dim vector, cache index expects "
                    f"{VECTOR_DIM} — skipping L2 write for this entry (L1 still cached)."
                )
                return None
            return struct.pack(f"{VECTOR_DIM}f", *vector)
        except Exception as e:
            logger.warning(f"Failed to embed query for cache: {e}")
            return None

    @staticmethod
    def _lexical_overlap(query_a: str, query_b: str) -> float:
        """Jaccard token overlap between two ALREADY-normalized query
        strings — the secondary gate get_semantic applies alongside dense
        cosine similarity (see that method's docstring for why). Simple
        on purpose, matching model_router.py's own stated philosophy for
        this codebase's heuristics: real mechanism, not a state-of-the-art
        classifier."""
        tokens_a, tokens_b = set(query_a.split()), set(query_b.split())
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    @staticmethod
    def _length_ratio(query_a: str, query_b: str) -> float:
        """Third gate (Phase 6 follow-up) — see SEMANTIC_CACHE_MIN_LENGTH_RATIO's
        config.py docstring for the live failure this catches that
        _lexical_overlap alone cannot (containment: a compound question's
        cached answer served for one of its own narrower clauses)."""
        len_a, len_b = len(query_a.split()), len(query_b.split())
        if len_a == 0 or len_b == 0:
            return 0.0
        return min(len_a, len_b) / max(len_a, len_b)

    # --- L2: semantic match ---

    async def get_semantic(self, corpus_id: str, query: str) -> dict[str, Any] | None:
        """
        KNN search over corpus_id-tagged vectors. Only meaningful for
        cache-eligible (RAG-sourced) entries — those are the only ones
        set_exact() ever writes a vector for, so a semantic hit is
        implicitly tool-gated already.

        Gated on vector similarity AND lexical token overlap (Phase 6
        follow-up) — dense cosine similarity alone is not sufficient.
        Live-observed false positive: "How is the Transformer different
        from LSTM?" vs. "Does the Transformer improve upon LSTM?" scored
        0.956 cosine similarity (bge-large-en-v1.5) despite being
        different questions — comparison vs. justification-of-improvement
        — that a human reader would not treat as equivalent. Raising the
        threshold would NOT have caught this specific case (0.956 clears
        any reasonable bar); the actual problem is that dense embeddings
        alone don't encode this distinction reliably. Consistent with
        this codebase's own existing answer to the same class of problem
        for document retrieval — hybrid dense+BM25, not dense-only — this
        applies the same principle to cache matching: a candidate KNN hit
        must ALSO share a minimum fraction of content words with the
        cached query before being served. Conservative on purpose: a
        false negative here just means a full pipeline run (correct,
        slightly slower); a false positive means silently serving the
        wrong answer.
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
            # COSINE distance in RediSearch is 1 - cosine_similarity, so
            # lower "score" is more similar — threshold check is inverted
            # relative to a plain similarity score.
            similarity = 1.0 - float(top.score)
            if similarity < settings.SEMANTIC_CACHE_THRESHOLD:
                return None

            matched_key = top.id.encode() if isinstance(top.id, str) else top.id
            raw_entries, raw_query = await client.hmget(matched_key, [b"entries", b"query"])
            if not raw_entries:
                return None

            # Lexical gate — see docstring. raw_query is absent for
            # entries written before this field existed (pre-Phase-6-fix);
            # treat that as a miss rather than either falsely passing or
            # crashing, since there's nothing to compare against.
            if not raw_query:
                logger.info("L2 candidate has no stored query text (pre-fix entry) — treating as miss")
                return None
            overlap = self._lexical_overlap(self._normalize(query), raw_query.decode())
            if overlap < settings.SEMANTIC_CACHE_LEXICAL_OVERLAP:
                logger.info(
                    f"L2 candidate rejected: similarity={similarity:.3f} cleared threshold "
                    f"but lexical_overlap={overlap:.2f} did not (< {settings.SEMANTIC_CACHE_LEXICAL_OVERLAP})"
                )
                return None

            length_ratio = self._length_ratio(self._normalize(query), raw_query.decode())
            if length_ratio < settings.SEMANTIC_CACHE_MIN_LENGTH_RATIO:
                logger.info(
                    f"L2 candidate rejected: similarity={similarity:.3f} and lexical_overlap={overlap:.2f} "
                    f"cleared their bars, but length_ratio={length_ratio:.2f} did not "
                    f"(< {settings.SEMANTIC_CACHE_MIN_LENGTH_RATIO}) — likely a narrower "
                    f"sub-question of a broader cached (possibly compound) question."
                )
                return None

            entries = json.loads(raw_entries)
            current_version = await self._get_corpus_version(corpus_id)
            for entry in entries:
                if entry["corpus_version"] == current_version:
                    logger.info(
                        f"L2 semantic cache hit (similarity={similarity:.3f}, "
                        f"lexical_overlap={overlap:.2f}, length_ratio={length_ratio:.2f})"
                    )
                    return entry
            return None

        except Exception as e:
            logger.warning(f"L2 semantic cache lookup failed: {e}")
            return None

    async def delete_by_corpus_id(self, corpus_id: str) -> int:
        """Cascade-delete hook for session expiry (session/cleanup.py) —
        closes the previously-known gap where Redis L1/L2 cache entries
        for an expired corpus_id were never cleaned up (unreachable after
        expiry, since corpus_id is server-derived and can't be presented
        again, but a genuine slow memory leak over long uptimes).
        SCAN (cursor-based, non-blocking), not KEYS — safe to run against
        a live Redis instance serving other requests concurrently. Works
        directly off the key space (see _l1_key's docstring for why
        corpus_id is a plaintext key segment) rather than the RediSearch
        index, so this doesn't depend on that index being healthy.
        Returns the number of keys deleted, for logging."""
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
        if self._redis:
            await self._redis.close()


redis_cache = RedisCache()

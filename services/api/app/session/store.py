import time
import uuid
from typing import List

from services.api.app.cache.redis import redis_client

# Single sorted set keyed by corpus_id, score = last-active unix epoch.
# A ZSET (not native Redis key TTL) is deliberate: purge_expired() below
# returns the list of ids it removed, giving later phases (3+) a hook to
# also delete the matching Qdrant points / MinIO objects
# when a session expires. Native EXPIRE has no such hook without enabling
# keyspace notifications.
ACTIVE_SESSIONS_KEY = "sessions:active"


class SessionStore:
    """Tracks corpus_id session liveness in Redis with a sliding-window TTL."""

    async def create_session(self) -> str:
        corpus_id = str(uuid.uuid4())
        client = redis_client.get_client()
        await client.zadd(ACTIVE_SESSIONS_KEY, {corpus_id: time.time()})
        return corpus_id

    async def touch(self, corpus_id: str) -> None:
        """Refresh last-active timestamp (sliding TTL). No-op if unknown."""
        client = redis_client.get_client()
        await client.zadd(ACTIVE_SESSIONS_KEY, {corpus_id: time.time()}, xx=True)

    async def is_valid(self, corpus_id: str, ttl_minutes: int) -> bool:
        client = redis_client.get_client()
        score = await client.zscore(ACTIVE_SESSIONS_KEY, corpus_id)
        if score is None:
            return False
        return (time.time() - score) < (ttl_minutes * 60)

    async def purge_expired(self, ttl_minutes: int) -> List[str]:
        """Remove sessions inactive past ttl_minutes. Returns purged ids so
        callers (future phases) can cascade-delete their own corpus-tagged
        data."""
        client = redis_client.get_client()
        cutoff = time.time() - (ttl_minutes * 60)
        expired = await client.zrangebyscore(ACTIVE_SESSIONS_KEY, min=0, max=cutoff)
        if expired:
            await client.zrem(ACTIVE_SESSIONS_KEY, *expired)
        return list(expired)


session_store = SessionStore()
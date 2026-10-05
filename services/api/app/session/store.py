"""
Redis-backed session store with a sliding-window TTL. Tracks which corpus_ids are active, and which have expired and need their data purged.
"""
import time
import uuid
from typing import List

from services.api.app.cache.redis import redis_client

ACTIVE_SESSIONS_KEY = "sessions:active"  # sorted set, score = last-active unix epoch


class SessionStore:
    """Tracks corpus_id session liveness in Redis with a sliding-window TTL."""

    async def create_session(self) -> str:
        """Creates a new session.

        Returns:
            The new corpus_id.
        """
        corpus_id = str(uuid.uuid4())
        client = redis_client.get_client()
        await client.zadd(ACTIVE_SESSIONS_KEY, {corpus_id: time.time()})
        return corpus_id

    async def touch(self, corpus_id: str) -> None:
        """Refreshes a session's last-active timestamp. No-op if unknown.

        Args:
            corpus_id: The session to refresh.
        """
        client = redis_client.get_client()
        await client.zadd(ACTIVE_SESSIONS_KEY, {corpus_id: time.time()}, xx=True)

    async def is_valid(self, corpus_id: str, ttl_minutes: int) -> bool:
        """Checks whether a session is still active.

        Args:
            corpus_id: The session to check.
            ttl_minutes: Inactivity window before a session is considered expired.

        Returns:
            True if the session exists and hasn't exceeded the TTL.
        """
        client = redis_client.get_client()
        score = await client.zscore(ACTIVE_SESSIONS_KEY, corpus_id)
        if score is None:
            return False
        return (time.time() - score) < (ttl_minutes * 60)

    async def purge_expired(self, ttl_minutes: int) -> List[str]:
        """Removes sessions inactive past the TTL.

        Args:
            ttl_minutes: Inactivity window before a session is considered expired.

        Returns:
            The purged corpus_ids, for cascade-deleting their data.
        """
        client = redis_client.get_client()
        cutoff = time.time() - (ttl_minutes * 60)
        expired = await client.zrangebyscore(ACTIVE_SESSIONS_KEY, min=0, max=cutoff)
        if expired:
            await client.zrem(ACTIVE_SESSIONS_KEY, *expired)
        return list(expired)


session_store = SessionStore()
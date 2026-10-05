"""
Singleton Redis connection used for session storage and health checks. The semantic cache (redis_cache.py) opens its own separate connection for raw binary vector storage.
"""
import redis.asyncio as redis
from services.api.app.config import settings


class RedisClient:
    """Singleton Redis connection pool for session storage and health checks."""

    def __init__(self):
        self.redis = None

    async def connect(self):
        """Opens the connection pool, if not already open."""
        if not self.redis:
            self.redis = redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )

    async def close(self):
        """Closes the connection pool."""
        if self.redis:
            await self.redis.close()

    def get_client(self):
        """Returns the active Redis client.

        Raises:
            RuntimeError: If connect() hasn't been called yet.
        """
        if self.redis is None:
            raise RuntimeError("Redis not connected. Call connect() first.")
        return self.redis


redis_client = RedisClient()
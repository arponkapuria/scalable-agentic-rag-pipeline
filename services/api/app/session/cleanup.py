"""
Background task that periodically purges expired sessions and cascades the deletion into every store tagged by corpus_id (Qdrant, Postgres, Redis cache, object storage).
"""
import asyncio
import logging

from services.api.app.config import settings
from services.api.app.session.store import session_store
from services.api.app.clients.qdrant import qdrant_client
from services.api.app.memory.postgres import postgres_memory
from services.api.app.cache.redis_cache import redis_cache
from libs.utils.s3_client import delete_corpus_objects

logger = logging.getLogger(__name__)


async def _cascade_delete(corpus_id: str) -> None:
    """Deletes every store's data for one expired corpus_id. Each store is try/excepted independently.

    Args:
        corpus_id: The expired session's corpus id.
    """
    try:
        await qdrant_client.delete_by_corpus_id(corpus_id)
    except Exception as e:
        logger.error(f"Cascade delete failed for Qdrant (corpus {corpus_id}): {e}")

    try:
        await postgres_memory.delete_by_corpus_id(corpus_id)
    except Exception as e:
        logger.error(f"Cascade delete failed for Postgres (corpus {corpus_id}): {e}")

    try:
        deleted = await redis_cache.delete_by_corpus_id(corpus_id)
        logger.info(f"Cascade delete: removed {deleted} cache entries for corpus {corpus_id}")
    except Exception as e:
        logger.error(f"Cascade delete failed for Redis cache (corpus {corpus_id}): {e}")

    try:
        deleted = await asyncio.to_thread(delete_corpus_objects, corpus_id)
        logger.info(f"Cascade delete: removed {deleted} S3 objects for corpus {corpus_id}")
    except Exception as e:
        logger.error(f"Cascade delete failed for S3/MinIO (corpus {corpus_id}): {e}")


async def _cleanup_loop() -> None:
    """Polls for expired sessions and cascade-deletes their data, forever, until cancelled."""
    poll_interval = max(60, settings.SESSION_TTL_MINUTES * 60 // 2)
    logger.info(f"Session cleanup loop started (poll_interval={poll_interval}s)")
    while True:
        try:
            expired = await session_store.purge_expired(settings.SESSION_TTL_MINUTES)
            if expired:
                logger.info(f"Purged {len(expired)} expired session(s): {expired}")
                for corpus_id in expired:
                    await _cascade_delete(corpus_id)
            else:
                logger.debug("Cleanup tick: no expired sessions")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Session cleanup loop error: {e}")
        await asyncio.sleep(poll_interval)


def start_cleanup_task() -> asyncio.Task:
    """Starts the cleanup loop as a background task.

    Returns:
        The created asyncio.Task, for cancellation at shutdown.
    """
    return asyncio.create_task(_cleanup_loop())


async def stop_cleanup_task(task: asyncio.Task) -> None:
    """Cancels the cleanup task and waits for it to stop.

    Args:
        task: The task returned by start_cleanup_task().
    """
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
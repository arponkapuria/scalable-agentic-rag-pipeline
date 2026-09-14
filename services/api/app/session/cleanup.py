import asyncio
import logging

from services.api.app.config import settings
from services.api.app.session.store import session_store
from services.api.app.clients.qdrant import qdrant_client
from services.api.app.memory.postgres import postgres_memory
from libs.utils.s3_client import delete_corpus_objects

logger = logging.getLogger(__name__)

# Run twice per TTL window (floor 1 min) so a session is never more than
# ~ttl/2 stale before it's purged, without polling so often it's wasted work.
_CHECK_INTERVAL_SECONDS = max(60, (settings.SESSION_TTL_MINUTES * 60) // 2)


async def _cascade_delete(corpus_id: str) -> None:
    """One expired corpus_id's full cleanup, across every store that
    tags data by it. Each store's failure is caught independently — a
    Qdrant hiccup shouldn't stop Postgres/MinIO cleanup for the same
    corpus, and one corpus's failure shouldn't stop the rest of this
    purge batch (see the caller's per-id try/except)."""
    await qdrant_client.delete_by_corpus_id(corpus_id)
    await postgres_memory.delete_by_corpus_id(corpus_id)
    deleted_objects = await asyncio.to_thread(delete_corpus_objects, corpus_id)
    logger.info(f"Cascade-deleted corpus {corpus_id}: Qdrant points, Postgres rows, {deleted_objects} MinIO object(s).")


async def _cleanup_loop() -> None:
    logger.info(f"[cleanup] Session cleanup loop started, checking every {_CHECK_INTERVAL_SECONDS}s (TTL={settings.SESSION_TTL_MINUTES}min).")
    while True:
        try:
            logger.info(f"[cleanup] tick — checking for sessions inactive > {settings.SESSION_TTL_MINUTES}min")
            purged = await session_store.purge_expired(settings.SESSION_TTL_MINUTES)
            if purged:
                logger.info(f"Purged {len(purged)} expired session(s): {purged}")
                for corpus_id in purged:
                    try:
                        await _cascade_delete(corpus_id)
                    except Exception as e:
                        # One corpus's cascade-delete failing (e.g. Qdrant
                        # briefly unreachable) shouldn't block cleanup for
                        # the rest of this batch, and the session record
                        # itself is already gone from Redis regardless —
                        # this only affects orphaned-data cleanup, not
                        # session validity.
                        logger.error(f"Cascade delete failed for corpus {corpus_id}: {e}", exc_info=True)
        except Exception as e:
            # Never let a transient Redis blip kill the loop.
            logger.error(f"Session cleanup pass failed: {e}", exc_info=True)

        await asyncio.sleep(_CHECK_INTERVAL_SECONDS)


def start_cleanup_task() -> asyncio.Task:
    task = asyncio.create_task(_cleanup_loop())
    logger.info("[cleanup] start_cleanup_task() called, task scheduled.")
    return task


async def stop_cleanup_task(task: asyncio.Task) -> None:
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
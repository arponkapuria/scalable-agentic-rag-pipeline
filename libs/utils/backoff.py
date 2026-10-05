"""
Decorator that retries an async HTTP call with exponential backoff and jitter, so repeated failures back off instead of hammering the server immediately.
"""
import logging
import functools
import random
import asyncio
import httpx

logger = logging.getLogger(__name__)


def exponential_backoff(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 10.0):
    """Decorator that retries an async function on httpx.HTTPError with exponential backoff and jitter.

    Args:
        max_retries: Number of retries before the exception is re-raised.
        base_delay: Base delay in seconds before the first retry.
        max_delay: Upper bound on the exponential delay (jitter is added on top).

    Returns:
        The decorator, wrapping the target async function.
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except httpx.HTTPError as e:
                    if retries >= max_retries:
                        logger.error(f"Max retries reached for {func.__name__}: {e}")
                        raise e

                    # delay = base * 2^retries, capped, plus random jitter to spread out concurrent retries
                    delay = min(base_delay * (2 ** retries), max_delay)
                    jitter = random.uniform(0, 0.5)
                    sleep_time = delay + jitter

                    logger.warning(f"Error in {func.__name__}: {e}. Retrying in {sleep_time:.2f}s...")
                    await asyncio.sleep(sleep_time)
                    retries += 1
        return wrapper
    return decorator
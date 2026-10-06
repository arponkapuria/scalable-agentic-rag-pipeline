"""
Inbound rate limiting for the open demo: fixed-window counters in Redis, enforced as FastAPI dependencies. Each limited route is capped per client IP and per session (corpus_id), since sessions are free to mint and IP alone would punish shared networks.
"""
import logging
import math
import time

from fastapi import Depends, HTTPException, Request
from redis.exceptions import RedisError

from services.api.app.cache.redis import redis_client
from services.api.app.config import settings
from services.api.app.session.dependency import get_corpus_id

logger = logging.getLogger(__name__)


def client_ip(request: Request) -> str:
    """Resolves the caller's IP, trusting X-Forwarded-For only behind our own proxy.

    Args:
        request: The incoming request.

    Returns:
        The client IP, or "unknown" if the socket has no peer address.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if settings.TRUSTED_PROXY and forwarded:
        # Rightmost entry is the one our proxy appended; anything left of it is client-supplied.
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


async def _enforce(scope: str, subject: str, limit: int, window_seconds: int) -> None:
    """Counts one hit against a fixed window and raises 429 once the limit is exceeded.

    Args:
        scope: Route name plus key type, e.g. "chat:ip".
        subject: The IP or corpus_id being limited.
        limit: Max hits allowed per window.
        window_seconds: Window length.

    Raises:
        HTTPException: 429 with a Retry-After header when over the limit.
    """
    now = time.time()
    # Window index in the key means each window gets a fresh counter, so EXPIRE can be set
    # unconditionally in the same transaction as INCR — no INCR/EXPIRE gap that could leave an immortal key.
    key = f"rl:{scope}:{subject}:{int(now // window_seconds)}"
    try:
        async with redis_client.get_client().pipeline(transaction=True) as pipe:
            pipe.incr(key)
            pipe.expire(key, window_seconds)
            count, _ = await pipe.execute()
    except (RedisError, RuntimeError) as e:
        # Fail open: sessions already need Redis, so an outage 401s every data route anyway.
        logger.warning(f"[rate_limit] Redis unavailable, allowing request scope={scope}: {e}")
        return

    if count > limit:
        retry_after = math.ceil(window_seconds - now % window_seconds)
        logger.info(f"[rate_limit] blocked scope={scope} subject={subject} count={count} limit={limit}")
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry in {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )


def limit_by_ip(route: str, limit: int, window_seconds: int):
    """Builds a dependency that rate-limits a route per client IP.

    Args:
        route: Route name used in the Redis key.
        limit: Max requests per window.
        window_seconds: Window length.

    Returns:
        An async FastAPI dependency.
    """
    async def dependency(request: Request) -> None:
        await _enforce(f"{route}:ip", client_ip(request), limit, window_seconds)

    return dependency


def limit_by_session(route: str, limit: int, window_seconds: int):
    """Builds a dependency that rate-limits a route per session (corpus_id).

    Depends on get_corpus_id so only validated sessions create counters — a forged cookie
    gets a 401 instead of a junk Redis key.

    Args:
        route: Route name used in the Redis key.
        limit: Max requests per window.
        window_seconds: Window length.

    Returns:
        An async FastAPI dependency.
    """
    async def dependency(corpus_id: str = Depends(get_corpus_id)) -> None:
        await _enforce(f"{route}:session", corpus_id, limit, window_seconds)

    return dependency
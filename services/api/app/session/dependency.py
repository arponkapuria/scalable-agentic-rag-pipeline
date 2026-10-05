"""
FastAPI dependency that derives corpus_id from the session cookie. Every data-touching route depends on this — corpus_id is never accepted from the client any other way.
"""
from fastapi import HTTPException, Request, status

from services.api.app.config import settings
from services.api.app.session.store import session_store

SESSION_COOKIE_NAME = "corpus_id"


async def get_corpus_id(request: Request) -> str:
    """Reads and validates the session cookie, refreshing its TTL on success.

    Args:
        request: The incoming request.

    Returns:
        The validated corpus_id.

    Raises:
        HTTPException: 401 if the cookie is missing or the session has expired.
    """
    corpus_id = request.cookies.get(SESSION_COOKIE_NAME)

    if not corpus_id or not await session_store.is_valid(corpus_id, settings.SESSION_TTL_MINUTES):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No active session. Call POST /api/v1/session/init first.",
        )

    await session_store.touch(corpus_id)
    return corpus_id
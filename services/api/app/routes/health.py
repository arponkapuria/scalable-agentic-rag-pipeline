"""
Health check endpoints: liveness (process is running) and readiness (dependencies are reachable).
"""
from fastapi import APIRouter, Response, status
from services.api.app.cache.redis import redis_client

router = APIRouter()


@router.get("/liveness")
async def liveness():
    """Returns 200 if the server process is running.

    Returns:
        A fixed status dict.
    """
    return {"status": "ok"}


@router.get("/readiness")
async def readiness(response: Response):
    """Checks connections to critical dependencies. Sets a 503 status if any are down.

    Args:
        response: The FastAPI response, whose status code is set on failure.

    Returns:
        A dict reporting each dependency's status.
    """
    status_report = {"redis": "down"}
    is_healthy = True

    try:
        r = redis_client.get_client()
        if await r.ping():
            status_report["redis"] = "up"
        else:
            is_healthy = False
    except Exception:
        is_healthy = False

    if not is_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return status_report
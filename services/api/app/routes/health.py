from fastapi import APIRouter, Response, status
from services.api.app.cache.redis import redis_client

router = APIRouter()

@router.get("/liveness")
async def liveness():
    """
    K8s Liveness Probe.
    Returns 200 if the server process is running.
    """
    return {"status": "ok"}

@router.get("/readiness")
async def readiness(response: Response):
    """
    K8s Readiness Probe.
    Checks connections to critical dependencies (Redis, DB).
    If this fails, K8s stops sending traffic to this pod.
    """
    status_report = {"redis": "down"}
    is_healthy = True

    # 1. Check Redis
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
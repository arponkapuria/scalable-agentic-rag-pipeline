"""
MinIO bucket-notification webhook: receives the same S3 event-record schema real AWS S3 uses, and dispatches one ingestion run per uploaded file as a background task.
"""
import logging
from urllib.parse import unquote_plus

from fastapi import APIRouter, BackgroundTasks, Request

from pipelines.ingestion.pipeline import run_ingestion

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/minio")
async def minio_event_webhook(request: Request, background_tasks: BackgroundTasks):
    """Receives MinIO bucket notification events and dispatches one ingestion run per file record.

    Args:
        request: The incoming webhook request, with the S3 event payload.
        background_tasks: FastAPI's background task queue — runs ingestion after the response.

    Returns:
        A fixed acceptance status.
    """
    event = await request.json()
    logger.info(f"MinIO webhook received: {event.get('EventName', 'unknown')}")

    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = unquote_plus(record["s3"]["object"]["key"])
        background_tasks.add_task(run_ingestion, bucket, key)

    return {"status": "accepted"}
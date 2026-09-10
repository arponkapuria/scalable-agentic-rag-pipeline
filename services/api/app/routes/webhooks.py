"""
MinIO's webhook notification target POSTs the same S3 event-record schema
AWS uses (Records[].s3.bucket.name / .object.key). Two backends can
handle it, picked by INGESTION_BACKEND (see PROGRESS.md carry-over notes
on deferring Ray):
  - "ray" (frozen, unmodified): s3_event_handler.py submits a Ray Job.
  - "in_process" (default): runs pipelines/ingestion/pipeline.py's
    run_ingestion() as a FastAPI BackgroundTask — not inline in this
    request handler, same "fast event handler, work happens after" shape
    the Ray path already had, just without a separate cluster.
"""
import logging
from urllib.parse import unquote_plus

from fastapi import APIRouter, BackgroundTasks, Request

from pipelines.jobs.s3_event_handler import handle_s3_event
from pipelines.ingestion.pipeline import run_ingestion
from services.api.app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/minio")
async def minio_event_webhook(request: Request, background_tasks: BackgroundTasks):
    """Receives MinIO bucket notification events, dispatches one ingestion
    run per file record."""
    event = await request.json()
    logger.info(f"MinIO webhook received: {event.get('EventName', 'unknown')}")

    if settings.INGESTION_BACKEND == "ray":
        handle_s3_event(event, context=None)
    else:
        for record in event.get("Records", []):
            bucket = record["s3"]["bucket"]["name"]
            key = unquote_plus(record["s3"]["object"]["key"])
            background_tasks.add_task(run_ingestion, bucket, key)

    return {"status": "accepted"}

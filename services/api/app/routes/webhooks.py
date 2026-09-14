"""
MinIO's webhook notification target POSTs the same S3 event-record schema
AWS uses (Records[].s3.bucket.name / .object.key). Ray's Job-Submission
ingestion path (pipelines/jobs/s3_event_handler.py) was removed along
with Neo4j — this project no longer needs a separate ingestion cluster;
Docling's parsing/chunking cost fits comfortably in-process on a single
FastAPI worker. run_ingestion() runs as a FastAPI BackgroundTask — fast
event handler, work happens after the response is sent.
"""
import logging
from urllib.parse import unquote_plus

from fastapi import APIRouter, BackgroundTasks, Request

from pipelines.ingestion.pipeline import run_ingestion

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/minio")
async def minio_event_webhook(request: Request, background_tasks: BackgroundTasks):
    """Receives MinIO bucket notification events, dispatches one ingestion
    run per file record."""
    event = await request.json()
    logger.info(f"MinIO webhook received: {event.get('EventName', 'unknown')}")

    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = unquote_plus(record["s3"]["object"]["key"])
        background_tasks.add_task(run_ingestion, bucket, key)

    return {"status": "accepted"}

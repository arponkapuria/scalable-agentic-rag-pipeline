"""Downloads the five eval papers and ingests them through the production ingestion pipeline.

Resumable per paper: papers already marked complete for the eval corpus are skipped.
"""
import asyncio
import logging
import uuid

import httpx

from libs.utils.s3_client import get_s3_client
from pipelines.ingestion.pipeline import run_ingestion
from services.api.app.clients.llm.mistral_client import mistral_client
from services.api.app.config import settings
from services.api.app.memory.models import Base
from services.api.app.memory.postgres import document_store, engine

from eval import config as eval_config

logger = logging.getLogger(__name__)


async def _download(url: str) -> bytes:
    """Downloads a file and returns its bytes."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content


async def _already_complete(corpus_id: str, filename: str) -> bool:
    """Returns True if this file is already fully ingested for the corpus."""
    docs = await document_store.list_by_corpus_id(corpus_id)
    return any(d.filename == filename and d.status == "complete" for d in docs)


async def _ingest_paper(paper: dict) -> None:
    """Downloads one paper, uploads it to S3, and runs production ingestion on it."""
    corpus_id = eval_config.EVAL_CORPUS_ID
    if await _already_complete(corpus_id, paper["filename"]):
        logger.info(f"[ingest-corpus] {paper['filename']} already complete — skipping")
        return

    logger.info(f"[ingest-corpus] downloading {paper['name']} from {paper['url']}")
    content = await _download(paper["url"])

    file_id = str(uuid.uuid4())
    s3_key = f"uploads/{corpus_id}/{file_id}.pdf"
    s3 = get_s3_client()
    await asyncio.to_thread(
        s3.put_object,
        Bucket=settings.S3_BUCKET_NAME,
        Key=s3_key,
        Body=content,
        ContentType="application/pdf",
        Metadata={"original_filename": paper["filename"], "corpus_id": corpus_id},
    )
    await document_store.create_pending(file_id, corpus_id, paper["filename"], s3_key)

    logger.info(f"[ingest-corpus] running production ingestion for {paper['filename']}")
    await run_ingestion(settings.S3_BUCKET_NAME, s3_key)
    logger.info(f"[ingest-corpus] {paper['filename']} complete")


async def run() -> None:
    """Creates the tables if missing, then ingests every paper in the eval config."""
    # The API server normally creates the tables, but this script runs without it.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await mistral_client.start()  # Needed for figure captioning during ingestion.
    try:
        for paper in eval_config.PAPERS:
            await _ingest_paper(paper)
    finally:
        await mistral_client.close()

    logger.info(f"[ingest-corpus] all papers processed for corpus_id={eval_config.EVAL_CORPUS_ID}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())
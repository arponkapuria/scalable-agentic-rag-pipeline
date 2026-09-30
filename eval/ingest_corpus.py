"""
Run 1 — downloads the 5 fixed arXiv papers, uploads each to MinIO/S3 at
the same uploads/{corpus_id}/{file_id}.ext key layout upload.py uses for a
real browser upload, then calls run_ingestion() directly (awaited, not as
a FastAPI BackgroundTask) — the exact same production ingestion path
(Docling parse/chunk -> embed -> index -> corpus_version bump), just
triggered from a script instead of a MinIO webhook. No ingestion code is
duplicated or reimplemented for eval.

Resumable per-paper: a paper already at status="complete" in the
Document table for EVAL_CORPUS_ID is skipped, so a partial run (e.g. one
paper hit a Mistral captioning issue mid-document) can just be re-run and
only the unfinished papers redo work.

Note: ingestion's own debug dump (INGEST_DEBUG_DUMP, chunks.json) writes
to a fixed logs/ingest_debug/{corpus_id}/chunks.json path per corpus_id,
overwritten on each paper — with 5 papers sharing one eval corpus_id,
only the LAST paper's dump survives. verify_gold.py therefore reads
straight from Qdrant (every paper's indexed chunks), not this dump.
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
    async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content


async def _already_complete(corpus_id: str, filename: str) -> bool:
    docs = await document_store.list_by_corpus_id(corpus_id)
    return any(d.filename == filename and d.status == "complete" for d in docs)


async def _ingest_paper(paper: dict) -> None:
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
    # Document table needs to exist — main.py's lifespan normally creates
    # it; this script runs standalone (no live API server needed), so it
    # does the same create_all() here.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await mistral_client.start()  # ingestion's figure captioning needs this
    try:
        for paper in eval_config.PAPERS:
            await _ingest_paper(paper)
    finally:
        await mistral_client.close()

    logger.info(f"[ingest-corpus] all papers processed for corpus_id={eval_config.EVAL_CORPUS_ID}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())
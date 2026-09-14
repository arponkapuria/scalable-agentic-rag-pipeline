"""
In-process ingestion driver — the only ingestion path now (Ray/Neo4j
removed entirely; project scope narrowed to ingestion + hybrid retrieval
+ reranking, no graph DB, no separate ingestion cluster).

Stages: fetch -> parse/chunk (Docling) -> embed (dense+sparse) ->
index into Qdrant -> bump corpus_version. Runs as a FastAPI BackgroundTask
triggered from the MinIO webhook route.

CPU-bound/blocking calls (Docling's layout/table-structure inference,
FastEmbed sparse encoding) are wrapped in asyncio.to_thread so they don't
block the event loop chat requests share with ingestion.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from services.api.app.clients.embedding import embedding_client
from services.api.app.config import settings
from libs.utils.s3_client import get_s3_client

from models.embeddings.fastembed_client import fastembed_client
from pipelines.ingestion.chunking.metadata import enrich_metadata
from pipelines.ingestion.indexing.qdrant import QdrantIndexer
from pipelines.ingestion.loaders import docling_loader
from pipelines.ingestion.debug_dump import dump_debug_artifact

logger = logging.getLogger(__name__)

# One QdrantIndexer for the process's whole lifetime (owns a DB
# connection) rather than rebuilt per ingestion job.
_qdrant_indexer: Optional[QdrantIndexer] = None


def get_qdrant_indexer() -> QdrantIndexer:
    global _qdrant_indexer
    if _qdrant_indexer is None:
        _qdrant_indexer = QdrantIndexer()
    return _qdrant_indexer


def _corpus_id_from_key(file_key: str) -> str:
    """Keys are uploads/{corpus_id}/{file_id}.ext, corpus_id derived from
    the object path itself, never client-supplied."""
    parts = file_key.split("/")
    if len(parts) < 2 or parts[0] != "uploads":
        raise ValueError(f"Unexpected key layout, can't extract corpus_id: {file_key}")
    return parts[1]


async def _fetch_object(bucket: str, file_key: str) -> bytes:
    """boto3 is sync — offloaded to a thread so it doesn't block the
    event loop chat requests are also running on."""
    def _get():
        client = get_s3_client()
        obj = client.get_object(Bucket=bucket, Key=file_key)
        return obj["Body"].read()

    return await asyncio.to_thread(_get)


async def _parse_and_chunk(content: bytes, filename: str, corpus_id: str) -> List[Dict[str, Any]]:
    """Format-agnostic — PDF/DOCX/HTML/MD all go through Docling's
    DocumentConverter + HybridChunker (loaders/docling_loader.py)."""
    chunks = await docling_loader.parse_and_chunk(content, filename, corpus_id)
    for chunk in chunks:
        chunk["metadata"].update(enrich_metadata(chunk["metadata"], chunk["text"]))
    return chunks


async def _embed(texts: List[str]) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
    dense = await embedding_client.embed_documents(texts)
    sparse = await asyncio.to_thread(fastembed_client.embed_sparse, texts)
    return dense, sparse


def _index_qdrant(chunks: List[Dict[str, Any]], dense: List[List[float]], sparse: List[Dict[str, Any]], corpus_id: str) -> None:
    batch = {
        "text": [c["text"] for c in chunks],
        "metadata": [c["metadata"] for c in chunks],
        "corpus_id": [corpus_id] * len(chunks),
        "dense_vector": dense,
        "sparse_vector": sparse,
    }
    get_qdrant_indexer()(batch)


def bump_corpus_version(corpus_id: str) -> None:
    """Sync Redis INCR — corpus_version increments when ingestion
    COMPLETES, not at upload start (locked design; powers Phase 5's
    before/after cache comparison)."""
    import redis as sync_redis

    client = sync_redis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        new_version = client.incr(f"corpus_version:{corpus_id}")
        logger.info(f"[ingest] corpus_version for {corpus_id} bumped to {new_version}")
    finally:
        client.close()


async def run_ingestion(bucket: str, file_key: str) -> None:
    """Entry point called from the MinIO webhook route as a FastAPI
    BackgroundTask."""
    corpus_id = _corpus_id_from_key(file_key)
    filename = file_key.rsplit("/", 1)[-1]
    logger.info(f"[ingest:{corpus_id}] stage=start file=s3://{bucket}/{file_key}")

    try:
        content = await _fetch_object(bucket, file_key)
        logger.info(f"[ingest:{corpus_id}] stage=fetch status=done bytes={len(content)}")

        chunks = await _parse_and_chunk(content, filename, corpus_id)
        logger.info(f"[ingest:{corpus_id}] stage=parse_chunk status=done chunks={len(chunks)}")
        dump_debug_artifact(corpus_id, "chunks", chunks)

        texts = [c["text"] for c in chunks]

        dense, sparse = await _embed(texts)
        logger.info(f"[ingest:{corpus_id}] stage=embed status=done vectors={len(dense)}")

        await asyncio.to_thread(_index_qdrant, chunks, dense, sparse, corpus_id)
        logger.info(f"[ingest:{corpus_id}] stage=index_qdrant status=done")

        await asyncio.to_thread(bump_corpus_version, corpus_id)
        logger.info(f"[ingest:{corpus_id}] stage=complete status=done")

    except Exception as e:
        logger.error(f"[ingest:{corpus_id}] stage=failed error={e}", exc_info=True)
        raise

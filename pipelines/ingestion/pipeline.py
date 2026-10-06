"""
In-process ingestion driver. Stages: fetch -> parse/chunk (Docling) -> embed (dense+sparse) -> index into Qdrant -> bump corpus_version. Runs as a FastAPI BackgroundTask triggered from the MinIO webhook route. CPU-bound/blocking calls (Docling inference, sparse encoding) run via asyncio.to_thread so they don't block the event loop chat requests share with ingestion.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from services.api.app.clients.embedding import embedding_client
from services.api.app.config import settings
from services.api.app.memory.postgres import document_store
from libs.utils.s3_client import get_s3_client

from services.api.app.clients.fastembed_client import fastembed_client
from pipelines.ingestion.chunking.metadata import enrich_metadata
from pipelines.ingestion.indexing.qdrant import QdrantIndexer
from pipelines.ingestion.loaders import docling_loader
from pipelines.ingestion.debug_dump import dump_debug_artifact

logger = logging.getLogger(__name__)

_qdrant_indexer: Optional[QdrantIndexer] = None  # one instance for the process lifetime, owns a DB connection


def get_qdrant_indexer() -> QdrantIndexer:
    global _qdrant_indexer
    if _qdrant_indexer is None:
        _qdrant_indexer = QdrantIndexer()
    return _qdrant_indexer


def _corpus_id_from_key(file_key: str) -> str:
    """Extracts corpus_id from an S3 key (uploads/{corpus_id}/{file_id}.ext) — never client-supplied.

    Args:
        file_key: The S3 object key.

    Returns:
        The corpus_id.

    Raises:
        ValueError: If the key doesn't match the expected layout.
    """
    parts = file_key.split("/")
    if len(parts) < 2 or parts[0] != "uploads":
        raise ValueError(f"Unexpected key layout, can't extract corpus_id: {file_key}")
    return parts[1]


class FileTooLargeError(ValueError):
    """Raised when an uploaded object exceeds FILE_UPLOAD_MAX_SIZE_MB."""


async def _fetch_object(bucket: str, file_key: str) -> bytes:
    """Fetches an object from S3/MinIO, rejecting (and deleting) it if over the size cap.

    The presign request's file_size is client-declared, so this stored-object check is the
    one that actually enforces the cap. boto3 is sync, offloaded to a thread.

    Args:
        bucket: The S3 bucket name.
        file_key: The object key.

    Returns:
        The object's raw bytes.

    Raises:
        FileTooLargeError: If the stored object exceeds the size cap.
    """
    max_bytes = settings.FILE_UPLOAD_MAX_SIZE_MB * 1024 * 1024

    def _get():
        client = get_s3_client()
        size = client.head_object(Bucket=bucket, Key=file_key)["ContentLength"]
        if size > max_bytes:
            client.delete_object(Bucket=bucket, Key=file_key)
            raise FileTooLargeError(f"File exceeds the {settings.FILE_UPLOAD_MAX_SIZE_MB} MB limit.")
        obj = client.get_object(Bucket=bucket, Key=file_key)
        return obj["Body"].read()

    return await asyncio.to_thread(_get)


async def _parse_and_chunk(content: bytes, filename: str, corpus_id: str) -> List[Dict[str, Any]]:
    """Parses and chunks a document via Docling, then enriches each chunk's metadata.

    Args:
        content: The raw file bytes.
        filename: The original filename, used for format detection and citations.
        corpus_id: The owning session's corpus id, for logging/debug artifacts.

    Returns:
        A list of chunk dicts, each with "text" and "metadata".
    """
    chunks = await docling_loader.parse_and_chunk(content, filename, corpus_id)
    for chunk in chunks:
        chunk["metadata"].update(enrich_metadata(chunk["metadata"], chunk["text"]))
    return chunks


async def _embed(texts: List[str]) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
    """Embeds a batch of chunk texts, dense and sparse.

    Args:
        texts: The chunk texts to embed.

    Returns:
        A (dense_vectors, sparse_vectors) tuple, same order as the input.
    """
    dense = await embedding_client.embed_documents(texts)
    sparse = await asyncio.to_thread(fastembed_client.embed_sparse, texts)
    return dense, sparse


def _index_qdrant(chunks: List[Dict[str, Any]], dense: List[List[float]], sparse: List[Dict[str, Any]], corpus_id: str) -> None:
    """Upserts one document's embedded chunks into Qdrant.

    Args:
        chunks: Chunk dicts with "text" and "metadata".
        dense: Dense embedding vectors, same order as chunks.
        sparse: Sparse embedding vectors, same order as chunks.
        corpus_id: The owning session's corpus id, tagged on every point.
    """
    batch = {
        "text": [c["text"] for c in chunks],
        "metadata": [c["metadata"] for c in chunks],
        "corpus_id": [corpus_id] * len(chunks),
        "dense_vector": dense,
        "sparse_vector": sparse,
    }
    get_qdrant_indexer()(batch)


def bump_corpus_version(corpus_id: str) -> None:
    """Increments the corpus_version counter on ingestion completion — powers the cache's before/after comparison.

    Args:
        corpus_id: The corpus whose version to bump.
    """
    import redis as sync_redis

    client = sync_redis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        new_version = client.incr(f"corpus_version:{corpus_id}")
        logger.info(f"[ingest] corpus_version for {corpus_id} bumped to {new_version}")
    finally:
        client.close()


def _file_id_from_key(file_key: str) -> str:
    """Extracts file_id (the key's filename stem) from an S3 key — matches Document.file_id, no extra data needed from the webhook payload.

    Args:
        file_key: The S3 object key.

    Returns:
        The file_id.
    """
    stem = file_key.rsplit("/", 1)[-1]
    return stem.rsplit(".", 1)[0] if "." in stem else stem


async def run_ingestion(bucket: str, file_key: str) -> None:
    """Runs the full ingestion pipeline for one uploaded file. Entry point for the MinIO webhook's background task.

    Args:
        bucket: The S3 bucket the file was uploaded to.
        file_key: The S3 object key.
    """
    corpus_id = _corpus_id_from_key(file_key)
    file_id = _file_id_from_key(file_key)
    logger.info(f"[ingest:{corpus_id}] stage=start file=s3://{bucket}/{file_key}")

    async def _set_status(status: str, error: str | None = None) -> None:
        # Best-effort — a DB hiccup here shouldn't abort ingestion, it just means ingest/status
        # temporarily lags reality until the next stage's write succeeds.
        try:
            await document_store.set_status(file_id, status, error)
        except Exception as e:
            logger.warning(f"[ingest:{corpus_id}] failed to update Document status={status}: {e}")

    # Real filename, not the S3 key stem — feeds directly into retriever.py's sources list and
    # the responder's "[Source: X]" citation.
    doc = await document_store.get_by_file_id(file_id)
    filename = doc.filename if doc else file_key.rsplit("/", 1)[-1]

    try:
        await _set_status("fetching")
        content = await _fetch_object(bucket, file_key)
        logger.info(f"[ingest:{corpus_id}] stage=fetch status=done bytes={len(content)}")

        await _set_status("parsing")
        chunks = await _parse_and_chunk(content, filename, corpus_id)
        logger.info(f"[ingest:{corpus_id}] stage=parse_chunk status=done chunks={len(chunks)}")
        dump_debug_artifact(corpus_id, "chunks", chunks)

        texts = [c["text"] for c in chunks]

        await _set_status("embedding")
        dense, sparse = await _embed(texts)
        logger.info(f"[ingest:{corpus_id}] stage=embed status=done vectors={len(dense)}")

        await _set_status("indexing")
        await asyncio.to_thread(_index_qdrant, chunks, dense, sparse, corpus_id)
        logger.info(f"[ingest:{corpus_id}] stage=index_qdrant status=done")

        await asyncio.to_thread(bump_corpus_version, corpus_id)
        await document_store.set_complete(file_id, chunk_count=len(chunks))
        logger.info(f"[ingest:{corpus_id}] stage=complete status=done")

    except Exception as e:
        logger.error(f"[ingest:{corpus_id}] stage=failed error={e}", exc_info=True)
        await _set_status("failed", error=str(e))
        raise
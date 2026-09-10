"""
Distributed Ray pipeline for parsing documents, generating embeddings and
knowledge graphs, and indexing them into Qdrant and Neo4j.

Runs as a Ray Job (submitted via s3_event_handler.py's JobSubmissionClient
call), one job per uploaded file — bucket/file_key come from sys.argv,
set by the MinIO webhook -> submit_ingestion_job() call chain.
"""
import asyncio
import logging
import os
import sys
from typing import Any, Dict

import pyarrow.fs
import ray

from pipelines.ingestion.embedding.compute import BatchEmbedder
from pipelines.ingestion.graph.extractor import GraphExtractor
from pipelines.ingestion.indexing.neo4j import Neo4jIndexer
from pipelines.ingestion.indexing.qdrant import QdrantIndexer

from pipelines.ingestion.chunking.metadata import enrich_metadata
from pipelines.ingestion.chunking.section_splitter import split_markdown_by_sections
from pipelines.ingestion.chunking.splitter import split_text
from pipelines.ingestion.loaders.opendataloader_pdf import parse_pdf_bytes
from libs.utils.document_parsing import parse_document
from services.api.app.config import settings

logger = logging.getLogger(__name__)

# Connects to the Ray cluster this job was submitted to — same call
# whether that's the local single-head/single-worker dev cluster or a
# real multi-node cluster later; the Job Submission API already puts this
# driver inside the target cluster's context.
ray.init(address="auto")


def _s3_filesystem():
    """pyarrow S3FileSystem, MinIO-aware via endpoint_override. None
    endpoint = real AWS, same as libs/utils/s3_client.py's boto3 pattern."""
    if settings.S3_ENDPOINT_URL:
        return pyarrow.fs.S3FileSystem(
            endpoint_override=settings.S3_ENDPOINT_URL,
            access_key=settings.AWS_ACCESS_KEY_ID,
            secret_key=settings.AWS_SECRET_ACCESS_KEY,
            scheme="http",
        )
    return pyarrow.fs.S3FileSystem(region=settings.AWS_REGION)


def _corpus_id_from_key(file_key: str) -> str:
    """Keys are laid out as uploads/{corpus_id}/{file_id}.ext by upload.py
    — corpus_id is the second path segment, never client-supplied here
    either (derived from the object path MinIO/S3 actually stored it at)."""
    parts = file_key.split("/")
    if len(parts) < 2 or parts[0] != "uploads":
        raise ValueError(f"Unexpected key layout, can't extract corpus_id: {file_key}")
    return parts[1]


def process_batch(batch: Dict[str, Any], corpus_id: str) -> Dict[str, Any]:
    """
    Ray Data transformation function. Receives a batch of file contents
    (bytes) and converts them into text chunks, tagged with corpus_id.
    PDF goes through OpenDataLoader + recursive section-based splitting;
    DOCX/HTML stay on the old flat loader + flat recursive splitter
    (Phase 3/4 scope is PDF-only — see opendataloader_pdf.py docstring).
    """
    results = []

    for i, content in enumerate(batch["bytes"]):
        filename = os.path.basename(batch["path"][i])
        ext = filename.lower().split(".")[-1]

        try:
            if ext == "pdf":
                # parse_pdf_bytes is now async (image captioning does a
                # real Gemini call) — this Ray worker function is called
                # synchronously with no loop already running, so
                # asyncio.run() per file is safe here (unlike the
                # in-process FastAPI pipeline, which already has one).
                markdown_text, metadata = asyncio.run(parse_pdf_bytes(content, filename))
                chunks = split_markdown_by_sections(markdown_text, chunk_size=512, overlap=50)
            else:
                raw_text, metadata = parse_document(content, filename)
                chunks = split_text(raw_text, chunk_size=512, overlap=50)

            for chunk in chunks:
                chunk["metadata"].update(enrich_metadata(metadata, chunk["text"]))
                chunk["corpus_id"] = corpus_id
                results.append(chunk)

        except Exception as e:
            logger.error(f"Failed processing file {filename}: {e}")

    return {
        "text": [r["text"] for r in results],
        "metadata": [r["metadata"] for r in results],
        "corpus_id": [r["corpus_id"] for r in results],
    }


def main(bucket_name: str, file_key: str):
    """One ingestion job per uploaded file (bucket, file_key) — not a
    whole-prefix batch job; the webhook fires per-object."""
    corpus_id = _corpus_id_from_key(file_key)
    logger.info(f"Starting ingestion for s3://{bucket_name}/{file_key} (corpus_id={corpus_id})")

    ds = ray.data.read_binary_files(
        paths=f"s3://{bucket_name}/{file_key}",
        include_paths=True,
        filesystem=_s3_filesystem(),
    )
    logger.info("File loaded from MinIO/S3")

    # 2. Parse & Chunk (Map Phase). No GPU locally — CPU-only throughout
    # (OpenDataLoader is JVM/rule-based, FastEmbed is CPU/ONNX — no GPU
    # anywhere in this pipeline).
    chunked_ds = ds.map_batches(
        process_batch,
        fn_kwargs={"corpus_id": corpus_id},
        batch_size=10,
        num_cpus=1,
    )
    logger.info("Parsing and chunking stage configured")

    # 3. FORK: Branch A — Embeddings (FastEmbed in-process, CPU).
    # compute=ActorPoolStrategy (not the deprecated concurrency= kwarg,
    # Ray 2.51+) is Ray Data's own actor-pool autoscaler — this is the
    # "min/max workers, capped low" mechanism from the locked design; on a
    # single-host local/demo cluster there's no literal node to add, so
    # autoscaling happens at the actor-pool level within available cluster
    # CPU, same mechanism a real multi-node cluster uses at a higher
    # ceiling. Fixed at size=1 here: each actor loads its own FastEmbed
    # model copy, and 2 concurrent copies is what caused an earlier OOM on
    # this capped container.
    vector_ds = chunked_ds.map_batches(
        BatchEmbedder,
        compute=ray.data.ActorPoolStrategy(size=1),
        batch_size=32,
    )
    logger.info("Embedding pipeline configured")

    # 4. FORK: Branch B — Graph Extraction (LLM, batched per Ray batch —
    # see graph/extractor.py's docstring for why this is now 1 call per
    # ~5 chunks instead of 1 call per chunk).
    graph_ds = chunked_ds.map_batches(
        GraphExtractor,
        compute=ray.data.ActorPoolStrategy(min_size=1, max_size=2),
        batch_size=settings.GRAPH_EXTRACTION_BATCH_SIZE,
    )
    logger.info("Graph extraction pipeline configured")

    # 5. Indexing (Write to DBs) — terminal sinks, same map_batches
    # callable-class pattern as BatchEmbedder/GraphExtractor.
    # .materialize() actually triggers execution; write_datasource() was
    # dead here before — QdrantIndexer/Neo4jIndexer were never
    # ray.data.Datasource subclasses, so that call path never ran.
    logger.info("Writing embeddings to Qdrant")
    vector_ds.map_batches(
        QdrantIndexer, compute=ray.data.ActorPoolStrategy(size=1), batch_size=100
    ).materialize()

    logger.info("Writing graph data to Neo4j")
    graph_ds.map_batches(
        Neo4jIndexer, compute=ray.data.ActorPoolStrategy(size=1), batch_size=50
    ).materialize()

    logger.info(f"Ingestion job completed successfully for corpus_id={corpus_id}.")

    # Phase 5: bump corpus_version now that ingestion has actually
    # completed (locked design: "increments when INGESTION of a new
    # document COMPLETES, not at upload start"). Cache lookups compare
    # against this to decide fresh-vs-stale, and to power the
    # before/after comparison view.
    _bump_corpus_version(corpus_id)


def _bump_corpus_version(corpus_id: str):
    """Sync Redis INCR — this runs at the end of a Ray job's driver
    process, not inside an async FastAPI request, so the sync redis client
    is simpler here than threading the async one through."""
    import redis as sync_redis

    client = sync_redis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        new_version = client.incr(f"corpus_version:{corpus_id}")
        logger.info(f"corpus_version for {corpus_id} bumped to {new_version}")
    finally:
        client.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

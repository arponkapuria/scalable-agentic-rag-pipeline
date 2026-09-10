"""
In-process ingestion driver — INGESTION_BACKEND=in_process (default, see
PROJECT_INSTRUCTIONS.md/PROGRESS.md carry-over notes on deferring Ray).

Runs the SAME stages, in the SAME order, using the SAME processing
classes (BatchEmbedder, GraphExtractor, QdrantIndexer, Neo4jIndexer) as
pipelines/ingestion/main.py's Ray Data path — just called directly from
this FastAPI-process coroutine instead of via ray.data.map_batches. Ray's
own driver stays frozen/untouched; re-enabling it later
(INGESTION_BACKEND=ray) is a config flip back to the existing path, not a
rewrite of either.

Reusing the app's single global `llm_client` (services.api.app.clients.
llm.factory) for GraphExtractor is what actually fixes the rate-limiter-
sharing bug: with everything in one process there's only one
BackendRateLimiter/CircuitBreaker per backend, shared by chat and
ingestion alike — correct, since the provider's quota is account-level
regardless of which logical component calls it.

CPU-bound/blocking calls (OpenDataLoader's JVM subprocess, PyMuPDF
cropping, FastEmbed sparse encoding) are wrapped in asyncio.to_thread so
they don't block the event loop chat requests share with ingestion.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

from services.api.app.clients.llm.factory import llm_client as shared_llm_client
from services.api.app.clients.embedding import embedding_client
from services.api.app.config import settings
from libs.utils.s3_client import get_s3_client

from models.embeddings.fastembed_client import fastembed_client
from pipelines.ingestion.chunking.metadata import enrich_metadata
from pipelines.ingestion.chunking.section_splitter import split_markdown_by_sections
from pipelines.ingestion.chunking.splitter import split_text
from pipelines.ingestion.graph.extractor import GraphExtractor
from pipelines.ingestion.indexing.neo4j import Neo4jIndexer
from pipelines.ingestion.indexing.qdrant import QdrantIndexer
from pipelines.ingestion.loaders.opendataloader_pdf import parse_pdf_bytes
from libs.utils.document_parsing import parse_document
from pipelines.ingestion.debug_dump import dump_debug_artifact

import asyncio

logger = logging.getLogger(__name__)

# One GraphExtractor for the process's whole lifetime, sharing the app's
# llm_client — mirrors the Ray actor's "build once in __init__" pattern,
# just without an actor. QdrantIndexer/Neo4jIndexer are similarly reused
# (they own a DB connection each) rather than rebuilt per ingestion job.
_graph_extractor: Optional[GraphExtractor] = None
_qdrant_indexer: Optional[QdrantIndexer] = None
_neo4j_indexer: Optional[Neo4jIndexer] = None


def _get_graph_extractor() -> GraphExtractor:
    global _graph_extractor
    if _graph_extractor is None:
        _graph_extractor = GraphExtractor(llm_client=shared_llm_client)
    return _graph_extractor


def _get_qdrant_indexer() -> QdrantIndexer:
    global _qdrant_indexer
    if _qdrant_indexer is None:
        _qdrant_indexer = QdrantIndexer()
    return _qdrant_indexer


def _get_neo4j_indexer() -> Neo4jIndexer:
    global _neo4j_indexer
    if _neo4j_indexer is None:
        _neo4j_indexer = Neo4jIndexer()
    return _neo4j_indexer


def _corpus_id_from_key(file_key: str) -> str:
    """Same layout as pipelines/ingestion/main.py's Ray path — keys are
    uploads/{corpus_id}/{file_id}.ext, corpus_id derived from the object
    path itself, never client-supplied."""
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
    ext = filename.lower().split(".")[-1]

    if ext == "pdf":
        # parse_pdf_bytes is already async (Gemini captioning call
        # inside) — but its OpenDataLoader/PyMuPDF work is still
        # synchronous/blocking; the captioning await points are the only
        # real yield points, which is an acceptable tradeoff here (one
        # PDF at a time, not a tight loop) rather than restructuring the
        # loader into fully non-blocking code for this phase.
        markdown_text, metadata = await parse_pdf_bytes(content, filename, corpus_id)
        chunks = split_markdown_by_sections(markdown_text, chunk_size=512, overlap=50)
    else:
        raw_text, metadata = await asyncio.to_thread(parse_document, content, filename)
        chunks = split_text(raw_text, chunk_size=512, overlap=50)

    for chunk in chunks:
        chunk["metadata"].update(enrich_metadata(metadata, chunk["text"]))

    return chunks


async def _embed(texts: List[str]) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
    dense = await embedding_client.embed_documents(texts)
    sparse = await asyncio.to_thread(fastembed_client.embed_sparse, texts)
    return dense, sparse


async def _extract_graph(texts: List[str], corpus_id: str) -> Tuple[List[list], List[list]]:
    """Chunks `texts` into GRAPH_EXTRACTION_BATCH_SIZE-sized groups and
    calls the shared GraphExtractor's extract_batch on each — same
    batching GraphExtractor.__call__ gets from Ray Data's batch_size, just
    driven directly instead of via map_batches."""
    extractor = _get_graph_extractor()
    batch_size = settings.GRAPH_EXTRACTION_BATCH_SIZE
    total_batches = (len(texts) + batch_size - 1) // batch_size

    all_nodes: List[list] = []
    all_edges: List[list] = []
    for batch_num, start in enumerate(range(0, len(texts), batch_size), start=1):
        group = texts[start:start + batch_size]
        results = await extractor.extract_batch(group)
        for nodes, edges in results:
            all_nodes.append(nodes)
            all_edges.append(edges)
        # Under sustained rate-limiting this stage is by far the slowest
        # and previously gave zero feedback between the start and final
        # "done" log — no way to tell "still working" from "hung."
        logger.info(f"[ingest:{corpus_id}] stage=graph_extract progress batch={batch_num}/{total_batches}")

    return all_nodes, all_edges


def _index_qdrant(chunks: List[Dict[str, Any]], dense: List[List[float]], sparse: List[Dict[str, Any]], corpus_id: str) -> None:
    batch = {
        "text": [c["text"] for c in chunks],
        "metadata": [c["metadata"] for c in chunks],
        "corpus_id": [corpus_id] * len(chunks),
        "dense_vector": dense,
        "sparse_vector": sparse,
    }
    _get_qdrant_indexer()(batch)


def _index_neo4j(graph_nodes: List[list], graph_edges: List[list], corpus_id: str) -> None:
    import json as _json
    batch = {
        "corpus_id": [corpus_id] * len(graph_nodes),
        "graph_nodes": [_json.dumps(n) for n in graph_nodes],
        "graph_edges": [_json.dumps(e) for e in graph_edges],
    }
    _get_neo4j_indexer()(batch)


def _bump_corpus_version(corpus_id: str) -> None:
    """Sync Redis INCR, matching pipelines/ingestion/main.py's Ray-path
    behavior — corpus_version increments when ingestion COMPLETES, not at
    upload start (locked design; powers Phase 5's before/after cache
    comparison)."""
    import redis as sync_redis

    client = sync_redis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        new_version = client.incr(f"corpus_version:{corpus_id}")
        logger.info(f"[ingest] corpus_version for {corpus_id} bumped to {new_version}")
    finally:
        client.close()


async def run_ingestion(bucket: str, file_key: str) -> None:
    """Entry point called from the MinIO webhook route as a FastAPI
    BackgroundTask when INGESTION_BACKEND=in_process."""
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

        graph_nodes, graph_edges = await _extract_graph(texts, corpus_id)
        node_count = sum(len(n) for n in graph_nodes)
        edge_count = sum(len(e) for e in graph_edges)
        logger.info(f"[ingest:{corpus_id}] stage=graph_extract status=done nodes={node_count} edges={edge_count}")
        dump_debug_artifact(corpus_id, "graph", [
            {"chunk_index": i, "text_preview": texts[i][:200], "nodes": graph_nodes[i], "edges": graph_edges[i]}
            for i in range(len(texts))
        ])

        await asyncio.to_thread(_index_qdrant, chunks, dense, sparse, corpus_id)
        logger.info(f"[ingest:{corpus_id}] stage=index_qdrant status=done")

        await asyncio.to_thread(_index_neo4j, graph_nodes, graph_edges, corpus_id)
        logger.info(f"[ingest:{corpus_id}] stage=index_neo4j status=done")

        await asyncio.to_thread(_bump_corpus_version, corpus_id)
        logger.info(f"[ingest:{corpus_id}] stage=complete status=done")

    except Exception as e:
        logger.error(f"[ingest:{corpus_id}] stage=failed error={e}", exc_info=True)
        raise

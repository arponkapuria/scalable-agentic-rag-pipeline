import asyncio
import logging
from typing import Dict

from services.api.app.agents.state import AgentState
from services.api.app.clients.embedding import embedding_client
from services.api.app.clients.qdrant import qdrant_client
from services.api.app.clients.reranker import reranker_client
from services.api.app.config import settings
from models.embeddings.fastembed_client import fastembed_client

logger = logging.getLogger(__name__)


async def retrieve_node(state: AgentState) -> Dict:
    """
    Hybrid retrieval, scoped to one corpus_id throughout:
    1. Embed the query (dense via FailoverEmbeddingClient, sparse via
       FastEmbed BM25).
    2. Qdrant hybrid search (dense+sparse, RRF-fused, corpus_id-filtered).
    3. Rerank the candidates, keep top N.

    Neo4j/graph search removed from this path (project scope narrowed to
    ingestion + hybrid retrieval + reranking — no graph DB) — this is now
    pure vector retrieval, no multi-hop/citation branch.
    """
    query = state["current_query"]
    corpus_id = state["corpus_id"]
    logger.info(f"[retrieve:{corpus_id}] stage=start query={query!r}")

    dense_vector, sparse_vector = await asyncio.gather(
        embedding_client.embed_query(query),
        asyncio.to_thread(fastembed_client.embed_sparse, [query]),
    )
    sparse_vector = sparse_vector[0]
    logger.info(f"[retrieve:{corpus_id}] stage=embed_query status=done")

    results = await qdrant_client.search_hybrid(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        corpus_id=corpus_id,
        limit=10,
        rrf_k=settings.RRF_K,
    )
    docs = []
    for r in results:
        payload = getattr(r, "payload", None)
        if not payload:
            continue
        docs.append(f"{payload.get('text', '')} [Source: {payload.get('filename', 'unknown')}]")

    logger.info(f"[retrieve:{corpus_id}] stage=hybrid_search status=done vector_hits={len(docs)}")

    if not docs:
        logger.info(f"[retrieve:{corpus_id}] stage=hybrid_search status=empty")
        return {"documents": [], "tool_used": "vector_search", "sources": []}

    scores = await reranker_client.rerank(query, docs)
    ranked = [doc for doc, _ in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)]
    final_docs = ranked[: settings.RERANK_TOP_N]
    logger.info(f"[retrieve:{corpus_id}] stage=rerank status=done candidates={len(docs)} kept={len(final_docs)}")

    # Sources parsed back out of the "[Source: filename]" suffix
    # appended above — cheap enough not to thread a separate structured-
    # sources channel through for this.
    sources = []
    for doc in final_docs:
        if "[Source: " in doc:
            sources.append(doc.split("[Source: ")[-1].rstrip("]"))

    logger.info(f"Retrieved {len(docs)} candidates, reranked to {len(final_docs)}.")
    return {"documents": final_docs, "tool_used": "vector_search", "sources": sources}

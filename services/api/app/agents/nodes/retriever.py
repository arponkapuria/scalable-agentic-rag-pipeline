import asyncio
import logging
from typing import Dict

from services.api.app.agents.state import AgentState
from services.api.app.clients.embedding import embedding_client
from services.api.app.clients.neo4j import neo4j_client
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
    3. Only query Neo4j if the corpus has more than one document — a
       single-paper corpus has no graph worth querying (locked design).
    4. Rerank the combined candidates, keep top N.
    """
    query = state["current_query"]
    corpus_id = state["corpus_id"]
    logger.info(f"[retrieve:{corpus_id}] stage=start query={query!r}")

    dense_vector, sparse_vector, doc_count = await asyncio.gather(
        embedding_client.embed_query(query),
        asyncio.to_thread(fastembed_client.embed_sparse, [query]),
        qdrant_client.count_distinct_documents(corpus_id),
    )
    sparse_vector = sparse_vector[0]
    logger.info(f"[retrieve:{corpus_id}] stage=embed_query status=done doc_count={doc_count}")

    async def run_vector_search():
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
        return docs

    async def run_graph_search():
        # Multi-hop/citation questions only make sense once there's more
        # than one document to relate — locked design's justification for
        # running Neo4j at all.
        if doc_count <= settings.GRAPH_SEARCH_MIN_DOCUMENTS - 1:
            logger.info(f"corpus={corpus_id} has {doc_count} document(s) — skipping graph search")
            return []
        cypher = """
        CALL db.index.fulltext.queryNodes("entity_index", $query) YIELD node, score
        WHERE node.corpus_id = $corpus_id
        MATCH (node)-[r]->(neighbor {corpus_id: $corpus_id})
        RETURN node.name + ' ' + type(r) + ' ' + neighbor.name as text
        LIMIT 5
        """
        try:
            results = await neo4j_client.query(cypher, {"query": query, "corpus_id": corpus_id})
            return [r["text"] for r in results]
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []

    vector_docs, graph_docs = await asyncio.gather(run_vector_search(), run_graph_search())
    logger.info(
        f"[retrieve:{corpus_id}] stage=hybrid_search status=done "
        f"vector_hits={len(vector_docs)} graph_hits={len(graph_docs)}"
    )

    seen = set()
    combined_docs = []
    for doc in graph_docs + vector_docs:  # graph first = higher priority pre-rerank
        if doc not in seen:
            seen.add(doc)
            combined_docs.append(doc)

    if not combined_docs:
        logger.info(f"[retrieve:{corpus_id}] stage=hybrid_search status=empty")
        return {"documents": [], "tool_used": "vector_search", "sources": []}

    scores = await reranker_client.rerank(query, combined_docs)
    ranked = [doc for doc, _ in sorted(zip(combined_docs, scores), key=lambda x: x[1], reverse=True)]
    final_docs = ranked[: settings.RERANK_TOP_N]
    logger.info(f"[retrieve:{corpus_id}] stage=rerank status=done candidates={len(combined_docs)} kept={len(final_docs)}")

    # Sources parsed back out of the "[Source: filename]" suffix
    # run_vector_search() appends to each doc string — cheap enough not to
    # thread a separate structured-sources channel through for this.
    sources = []
    for doc in final_docs:
        if "[Source: " in doc:
            sources.append(doc.split("[Source: ")[-1].rstrip("]"))

    logger.info(f"Retrieved {len(combined_docs)} candidates, reranked to {len(final_docs)}.")
    # tool_used marks this as RAG-sourced — Phase 5's L2 semantic cache
    # only applies to vector_search/graph_search answers, never
    # sandbox/web_search (locked design: staleness/precision risk there).
    tool_used = "vector_search+graph_search" if graph_docs else "vector_search"
    return {"documents": final_docs, "tool_used": tool_used, "sources": sources}

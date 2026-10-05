"""
Vector search tool: lets the agent explicitly invoke hybrid retrieval as a tool call, using the same dense+sparse/RRF path the retriever node uses for the default retrieval flow.
"""
import asyncio

from services.api.app.clients.fastembed_client import fastembed_client
from services.api.app.clients.embedding import embedding_client
from services.api.app.clients.qdrant import qdrant_client


async def search_vector_tool(query: str, corpus_id: str) -> str:
    """Searches the vector database, scoped to one corpus_id.

    Args:
        query: The search query.
        corpus_id: Restricts results to this corpus only — never cross-tenant.

    Returns:
        Formatted search results, or a message if none were found or the search failed.
    """
    try:
        dense_vector, sparse_vector = await asyncio.gather(
            embedding_client.embed_query(query),
            asyncio.to_thread(fastembed_client.embed_sparse, [query]),
        )
        results = await qdrant_client.search_hybrid(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector[0],
            corpus_id=corpus_id,
            limit=3,
        )

        if not results:
            return "No relevant documents found."

        formatted = ""
        for r in results:
            payload = r.payload or {}
            filename = payload.get("filename", "Unknown")
            page = payload.get("page", "N/A")
            formatted += f"- Content: {payload.get('text', '')[:200]}... [Source: {filename}, Page: {page}]\n"

        return formatted

    except Exception as e:
        return f"Search Error: {str(e)}"
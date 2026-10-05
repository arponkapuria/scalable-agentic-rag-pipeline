"""
Async Qdrant client: hybrid dense+sparse collection, hybrid/dense-only/sparse-only search, and corpus-scoped deletion.
"""
from qdrant_client import AsyncQdrantClient, models
from services.api.app.config import settings


class VectorDBClient:
    """Async client for Qdrant."""

    def __init__(self):
        self.client = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            prefer_grpc=False,
        )
        self._collections_ready = False

    async def init_collections(self):
        """Creates the main collection if it doesn't exist yet. Idempotent, lazy."""
        if self._collections_ready:
            return

        collections = await self.client.get_collections()
        existing = {c.name for c in collections.collections}

        if settings.QDRANT_COLLECTION not in existing:
            await self.client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config={"dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)},
                sparse_vectors_config={"sparse": models.SparseVectorParams()},
            )

        self._collections_ready = True

    async def search_hybrid(
        self,
        dense_vector: list[float],
        sparse_vector: dict,
        corpus_id: str,
        limit: int = 10,
        rrf_k: int = 60,
    ):
        """Searches dense and sparse vectors in parallel, fused with RRF, filtered to one corpus_id.

        Args:
            dense_vector: Query embedding for the dense index.
            sparse_vector: Query sparse vector, as {"indices": [...], "values": [...]}.
            corpus_id: Restricts results to this corpus only.
            limit: Max results to return.
            rrf_k: Not currently forwarded into the fusion query — Qdrant uses its own default.

        Returns:
            The matching points, with payload.
        """
        await self.init_collections()
        response = await self.client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            prefetch=[
                models.Prefetch(query=dense_vector, using="dense", limit=limit),
                models.Prefetch(
                    query=models.SparseVector(indices=sparse_vector["indices"], values=sparse_vector["values"]),
                    using="sparse",
                    limit=limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=models.Filter(
                must=[models.FieldCondition(key="corpus_id", match=models.MatchValue(value=corpus_id))]
            ),
            limit=limit,
            with_payload=True,
        )
        return response.points

    async def search_dense(self, dense_vector: list[float], corpus_id: str, limit: int = 10):
        """Dense-only search — used by the evaluation harness's retrieval ablation.

        Args:
            dense_vector: Query embedding.
            corpus_id: Restricts results to this corpus only.
            limit: Max results to return.

        Returns:
            The matching points, with payload.
        """
        await self.init_collections()
        response = await self.client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=dense_vector,
            using="dense",
            query_filter=models.Filter(
                must=[models.FieldCondition(key="corpus_id", match=models.MatchValue(value=corpus_id))]
            ),
            limit=limit,
            with_payload=True,
        )
        return response.points

    async def search_sparse(self, sparse_vector: dict, corpus_id: str, limit: int = 10):
        """BM25-only search — same evaluation-ablation use case as search_dense.

        Args:
            sparse_vector: Query sparse vector, as {"indices": [...], "values": [...]}.
            corpus_id: Restricts results to this corpus only.
            limit: Max results to return.

        Returns:
            The matching points, with payload.
        """
        await self.init_collections()
        response = await self.client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=models.SparseVector(indices=sparse_vector["indices"], values=sparse_vector["values"]),
            using="sparse",
            query_filter=models.Filter(
                must=[models.FieldCondition(key="corpus_id", match=models.MatchValue(value=corpus_id))]
            ),
            limit=limit,
            with_payload=True,
        )
        return response.points

    async def delete_by_corpus_id(self, corpus_id: str) -> None:
        """Deletes every point tagged with the given corpus_id.

        Args:
            corpus_id: The corpus whose points should be removed.
        """
        await self.init_collections()
        await self.client.delete(
            collection_name=settings.QDRANT_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="corpus_id", match=models.MatchValue(value=corpus_id))]
                )
            ),
        )

    async def close(self):
        """Closes the Qdrant connection."""
        await self.client.close()


qdrant_client = VectorDBClient()
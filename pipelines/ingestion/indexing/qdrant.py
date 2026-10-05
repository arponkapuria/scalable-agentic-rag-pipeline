"""
Terminal write step of the ingestion pipeline — upserts embedded chunks into Qdrant. A separate, synchronous client from services/api/app/clients/qdrant.py's async one, called via asyncio.to_thread from pipeline.py.
"""
import os
import uuid
from typing import Any, Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models


class QdrantIndexer:
    """Batches are column-oriented dicts (dict of lists), matching the rest of the pipeline's batch shape."""

    def __init__(self):
        host = os.getenv("QDRANT_HOST", "qdrant-service")
        port = int(os.getenv("QDRANT_PORT", 6333))
        self.collection_name = os.getenv("QDRANT_COLLECTION", "omnirag_collection")
        self.client = QdrantClient(host=host, port=port)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Creates the collection if it doesn't exist yet. Schema matches services/api/app/clients/qdrant.py's async client exactly (dense 1024-dim cosine + sparse) — a mismatch would silently break search instead of erroring."""
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={"dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)},
                sparse_vectors_config={"sparse": models.SparseVectorParams()},
            )

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Upserts one batch of embedded chunks.

        Args:
            batch: Column-oriented dict with "text", "metadata", "corpus_id", "dense_vector",
                and optionally "sparse_vector" — every entry a list, same length.

        Returns:
            The same batch, unmodified.
        """
        points = []
        n = len(batch.get("text", []))

        for i in range(n):
            if "dense_vector" not in batch or i >= len(batch["dense_vector"]):
                continue

            metadata = batch["metadata"][i]
            payload = {
                "text": batch["text"][i],
                "filename": metadata.get("filename"),
                "page": metadata.get("page", 0),
                "section": metadata.get("section", "Document"),
                "corpus_id": batch["corpus_id"][i],  # every query-time search filters on this — a wrong value here is a cross-tenant leak
            }

            vector = {"dense": batch["dense_vector"][i]}
            if "sparse_vector" in batch:
                sparse = batch["sparse_vector"][i]
                vector["sparse"] = models.SparseVector(indices=sparse["indices"], values=sparse["values"])

            points.append(models.PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))

        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

        return batch
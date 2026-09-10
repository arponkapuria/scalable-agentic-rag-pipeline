import os
import uuid
from typing import Any, Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models


class QdrantIndexer:
    """
    Ray Data terminal sink — same callable-class shape as BatchEmbedder/
    GraphExtractor (map_batches, not write_datasource: QdrantIndexer isn't
    a ray.data.Datasource, and the old write_datasource() call was dead —
    never actually executes any Ray write path). Batches are column-
    oriented dicts (dict of lists), matching what BatchEmbedder produces,
    not row-lists.
    """
    def __init__(self):
        host = os.getenv("QDRANT_HOST", "qdrant-service")   # qdrant-service = internal K8s DNS
        port = int(os.getenv("QDRANT_PORT", 6333))
        self.collection_name = os.getenv("QDRANT_COLLECTION", "omnirag_collection")
        self.client = QdrantClient(host=host, port=port)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Ingestion has to be able to create the collection itself, not
        just assume it exists — previously this indexer just upserted
        straight in, which worked as long as nothing ever deleted the
        collection, but 404'd hard (after graph extraction had already
        run, wasting that work) the moment it didn't exist. Schema
        mirrors clients/qdrant.py's async init_collections() exactly
        (dense 1024-dim cosine + sparse) — retrieval assumes this exact
        shape, so ingestion creating a different one would silently break
        search instead of erroring."""
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={"dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)},
                sparse_vectors_config={"sparse": models.SparseVectorParams()},
            )

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Every point carries corpus_id — every query-time search filters
        on this field, so a missing/wrong value here is a cross-tenant
        data leak, not just a bad chunk."""
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
                "corpus_id": batch["corpus_id"][i],
            }

            vector = {"dense": batch["dense_vector"][i]}
            if "sparse_vector" in batch:
                sparse = batch["sparse_vector"][i]
                vector["sparse"] = models.SparseVector(indices=sparse["indices"], values=sparse["values"])

            points.append(models.PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))

        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

        return batch

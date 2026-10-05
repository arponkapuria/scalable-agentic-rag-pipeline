"""
Shared dense + sparse embedding client — FastEmbed, ONNX/CPU, $0, no GPU. Used in-process by both query-time embedding (chat, cache) and document-time embedding (ingestion) — one implementation, not two paths to keep in sync.

Sync by design (FastEmbed itself is sync/CPU-bound) — call via asyncio.to_thread from async callers.
"""
from fastembed import TextEmbedding, SparseTextEmbedding
from services.api.app.config import settings


class FastEmbedClient:
    def __init__(self):
        self._dense: TextEmbedding | None = None
        self._sparse: SparseTextEmbedding | None = None

    def _dense_model(self) -> TextEmbedding:
        if self._dense is None:
            self._dense = TextEmbedding(model_name=settings.FASTEMBED_MODEL, cache_dir=settings.FASTEMBED_CACHE_DIR)
        return self._dense

    def _sparse_model(self) -> SparseTextEmbedding:
        if self._sparse is None:
            # Qdrant/bm25 — feeds Qdrant's native sparse-vector index directly.
            self._sparse = SparseTextEmbedding(model_name="Qdrant/bm25", cache_dir=settings.FASTEMBED_CACHE_DIR)
        return self._sparse

    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._dense_model().embed(texts)]

    def embed_sparse(self, texts: list[str]) -> list[dict]:
        """Qdrant-ready sparse vectors: [{"indices": [...], "values": [...]}, ...]."""
        return [
            {"indices": v.indices.tolist(), "values": v.values.tolist()}
            for v in self._sparse_model().embed(texts)
        ]


fastembed_client = FastEmbedClient()
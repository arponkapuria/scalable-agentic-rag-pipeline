"""
Shared dense + sparse embedding client — FastEmbed, ONNX/CPU, $0, no GPU.
Same class is used in-process by both the FastAPI app (query-time
embedding) and the Ray ingestion worker (document-time embedding) — one
implementation, not two paths to keep in sync (locked design: "same code,
both local and cloud").

Sync by design (FastEmbed itself is sync/CPU-bound) — call directly from
Ray workers, or via asyncio.to_thread from FastAPI request handlers.
"""
from fastembed import TextEmbedding, SparseTextEmbedding
from services.api.app.config import settings


class FastEmbedClient:
    # Models are lazy-loaded (ONNX session init is not free) and cached on
    # first use, not at import time — keeps module import cheap for code
    # paths that only need one of dense/sparse.
    def __init__(self):
        self._dense: TextEmbedding | None = None
        self._sparse: SparseTextEmbedding | None = None

    def _dense_model(self) -> TextEmbedding:
        if self._dense is None:
            self._dense = TextEmbedding(model_name=settings.FASTEMBED_MODEL, cache_dir=settings.FASTEMBED_CACHE_DIR)
        return self._dense

    def _sparse_model(self) -> SparseTextEmbedding:
        if self._sparse is None:
            # Qdrant/bm25 — FastEmbed's BM25 sparse export, feeds Qdrant's
            # native sparse-vector index directly (locked design: BM25
            # sparse + dense, fused via RRF).
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

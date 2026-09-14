"""
Local reranker — FastEmbed's bge-reranker-base cross-encoder, ONNX/CPU, $0.
Slots in after RRF fusion (locked design). Sync by design, same
call-from-Ray-or-via-to_thread pattern as fastembed_client.py.
"""
from fastembed.rerank.cross_encoder import TextCrossEncoder
from services.api.app.config import settings


class FastEmbedReranker:
    def __init__(self):
        self._model: TextCrossEncoder | None = None

    def _get_model(self) -> TextCrossEncoder:
        if self._model is None:
            self._model = TextCrossEncoder(model_name=settings.FASTEMBED_RERANKER_MODEL, cache_dir=settings.FASTEMBED_CACHE_DIR)
        return self._model

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Returns relevance scores in the same order as `documents` (higher = more relevant)."""
        return list(self._get_model().rerank(query, documents))


fastembed_reranker = FastEmbedReranker()

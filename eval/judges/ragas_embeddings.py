"""
Wraps the app's own FastEmbed embedding_client as ragas's modern
BaseRagasEmbedding interface (embed_text/aembed_text) — the "local
embedder, zero API cost" piece EVALUATION_DESIGN.md calls for. Used by
ragas's AnswerRelevancy (embeds the original question + its own
reverse-generated question, cosine similarity) and AnswerCorrectness
(embeds the answer + reference for the semantic-similarity component of
its weighted score).
"""
from eval import _ragas_compat  # noqa: F401 — import-order fix, see that module
from ragas.embeddings.base import BaseRagasEmbedding

from services.api.app.clients.embedding import embedding_client


class FastEmbedRagasEmbedding(BaseRagasEmbedding):
    """Only the async path is implemented — every metric in this project
    that touches embeddings calls aembed_text/aembed_texts exclusively
    (confirmed by reading ragas.metrics.collections' source), and
    embedding_client itself is async (FastEmbed's ONNX inference runs in
    a thread under the hood — see clients/embedding.py)."""

    def embed_text(self, text: str, **kwargs) -> list[float]:
        raise NotImplementedError(
            "sync embed_text not supported — this project's FastEmbed "
            "client and every eval script here are async-only."
        )

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return await embedding_client.embed_query(text)


fastembed_ragas_embedding = FastEmbedRagasEmbedding()
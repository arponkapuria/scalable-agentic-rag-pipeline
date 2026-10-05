"""Adapter that lets ragas use the app's local FastEmbed client for embeddings."""
from eval import _ragas_compat  # noqa: F401 — must load before ragas
from ragas.embeddings.base import BaseRagasEmbedding

from services.api.app.clients.embedding import embedding_client


class FastEmbedRagasEmbedding(BaseRagasEmbedding):
    """ragas embedding model backed by the local embedding client, async only."""

    def embed_text(self, text: str, **kwargs) -> list[float]:
        """Not supported, use aembed_text instead."""
        raise NotImplementedError(
            "sync embed_text not supported — this project's FastEmbed "
            "client and every eval script here are async-only."
        )

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        """Returns the embedding vector for the given text."""
        return await embedding_client.embed_query(text)


fastembed_ragas_embedding = FastEmbedRagasEmbedding()
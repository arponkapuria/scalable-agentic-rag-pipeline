"""
Run 2 — 4-way retrieval ablation per question: dense-only, BM25-only,
hybrid (RRF-fused), hybrid+rerank. The last is exactly what the shipped
pipeline (retriever.py) hands the responder — everything else exists
purely to isolate each retrieval component's own contribution, per
EVALUATION_DESIGN.md's ablation row.

Resumable: skips any variant already saved for a question. Zero LLM
calls — pure Qdrant + local FastEmbed + local reranker, so re-running
this costs nothing but wall-clock time (a few seconds per question).
"""
import asyncio
import logging

from models.embeddings.fastembed_client import fastembed_client
from services.api.app.clients.embedding import embedding_client
from services.api.app.clients.qdrant import qdrant_client
from services.api.app.clients.reranker import reranker_client
from services.api.app.config import settings

from eval import config as eval_config
from eval import storage

logger = logging.getLogger(__name__)

VARIANTS = ("dense_only", "bm25_only", "hybrid", "hybrid_rerank")


def _point_to_dict(point, score: float) -> dict:
    payload = getattr(point, "payload", None) or {}
    return {
        "text": payload.get("text", ""),
        "filename": payload.get("filename", "unknown"),
        "page": payload.get("page", 0),
        "score": score,
    }


async def _retrieve_one(question: dict) -> dict:
    qid = question["id"]
    query = question["question"]

    if storage.has_keys("retrieval", eval_config.RESULTS_DIR, qid, list(VARIANTS)):
        logger.info(f"[retrieve:{qid}] all variants already present — skipping")
        return storage.load(storage.question_path("retrieval", eval_config.RESULTS_DIR, qid))

    dense_vector, sparse_vectors = await asyncio.gather(
        embedding_client.embed_query(query),
        asyncio.to_thread(fastembed_client.embed_sparse, [query]),
    )
    sparse_vector = sparse_vectors[0]

    dense_points, sparse_points, hybrid_points = await asyncio.gather(
        qdrant_client.search_dense(dense_vector, eval_config.EVAL_CORPUS_ID, limit=eval_config.RETRIEVAL_ABLATION_LIMIT),
        qdrant_client.search_sparse(sparse_vector, eval_config.EVAL_CORPUS_ID, limit=eval_config.RETRIEVAL_ABLATION_LIMIT),
        qdrant_client.search_hybrid(
            dense_vector, sparse_vector, eval_config.EVAL_CORPUS_ID,
            limit=eval_config.RETRIEVAL_ABLATION_LIMIT, rrf_k=settings.RRF_K,
        ),
    )

    dense_only = [_point_to_dict(p, p.score) for p in dense_points]
    bm25_only = [_point_to_dict(p, p.score) for p in sparse_points]
    hybrid = [_point_to_dict(p, p.score) for p in hybrid_points]

    hybrid_rerank: list[dict] = []
    if hybrid:
        texts = [d["text"] for d in hybrid]
        scores = await reranker_client.rerank(query, texts)
        ranked = sorted(zip(hybrid, scores), key=lambda x: x[1], reverse=True)
        hybrid_rerank = [
            {**doc, "score": float(score)} for doc, score in ranked[: settings.RERANK_TOP_N]
        ]

    updates = {
        "dense_only": dense_only,
        "bm25_only": bm25_only,
        "hybrid": hybrid,
        "hybrid_rerank": hybrid_rerank,
    }
    logger.info(
        f"[retrieve:{qid}] dense={len(dense_only)} bm25={len(bm25_only)} "
        f"hybrid={len(hybrid)} hybrid_rerank={len(hybrid_rerank)}"
    )
    return storage.update("retrieval", eval_config.RESULTS_DIR, qid, updates)


async def run() -> None:
    data = storage.load(eval_config.DATASET_PATH)
    questions = data["questions"]
    for q in questions:
        await _retrieve_one(q)
    logger.info(f"[retrieve] ablation complete for {len(questions)} questions.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())
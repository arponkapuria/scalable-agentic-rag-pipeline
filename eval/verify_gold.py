"""
Human-in-the-loop checkpoint (EVALUATION_DESIGN.md): after Run 1 (ingest),
before Run 2 (retrieve) — confirms every question's gold_span was
actually found in SOME indexed chunk for the eval corpus. A wrong or
missing gold span silently breaks Hit@5/MRR/Precision@5 for that question
forever (it scores 0 not because retrieval failed, but because the
ground truth never matched anything) — this has to run before spending
any retrieval/generation calls on a broken question.

Reads straight from Qdrant (every indexed chunk for EVAL_CORPUS_ID, via
scroll), not the ingestion debug dump — with 5 papers sharing one eval
corpus_id, the debug dump (logs/ingest_debug/{corpus_id}/chunks.json)
gets overwritten by each paper in turn, so only the last paper's chunks
would survive there. Qdrant itself has everything.
"""
import asyncio
import logging

from qdrant_client import models

from services.api.app.clients.qdrant import qdrant_client
from services.api.app.config import settings

from eval import config as eval_config
from eval import storage
from eval.metrics.deterministic import normalize_text

logger = logging.getLogger(__name__)


async def _load_all_chunk_texts() -> list[str]:
    await qdrant_client.init_collections()
    texts: list[str] = []
    next_offset = None
    while True:
        points, next_offset = await qdrant_client.client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(
                    key="corpus_id", match=models.MatchValue(value=eval_config.EVAL_CORPUS_ID)
                )]
            ),
            limit=200,
            offset=next_offset,
            with_payload=True,
        )
        texts.extend(p.payload.get("text", "") for p in points)
        if next_offset is None:
            break
    return texts


async def main() -> None:
    data = storage.load(eval_config.DATASET_PATH)
    questions = data["questions"]
    chunk_texts = [normalize_text(t) for t in await _load_all_chunk_texts()]

    if not chunk_texts:
        print(f"FAIL: no indexed chunks found for corpus_id={eval_config.EVAL_CORPUS_ID}. "
              "Run `python -m eval.cli ingest` first.")
        return

    missing = []
    for q in questions:
        gold_span = q.get("gold_span")
        if not gold_span:
            continue  # unanswerable questions have no gold span by design
        needle = normalize_text(gold_span)
        if not any(needle in text for text in chunk_texts):
            missing.append(q["id"])

    total_answerable = sum(1 for q in questions if q.get("gold_span"))
    if missing:
        print(f"FAIL: {len(missing)}/{total_answerable} gold spans NOT found in any indexed chunk:")
        for qid in missing:
            print(f"  - {qid}")
        print(f"\n{len(chunk_texts)} chunks indexed total. Fix questions.json (adjust gold_span "
              "to match the real chunk text, verbatim substring) before running Run 2.")
    else:
        print(f"OK: all {total_answerable} gold spans found across {len(chunk_texts)} indexed chunks.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
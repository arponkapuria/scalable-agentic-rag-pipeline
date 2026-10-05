"""Checks that every answerable question's gold span appears in an indexed chunk.

Run after ingestion and before retrieval, since a missing span would score that question 0 for the wrong reason.
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
    """Returns the text of every chunk indexed for the eval corpus, read straight from Qdrant."""
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
    """Prints OK, or lists the question ids whose gold span is missing from every chunk."""
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
            continue  # Unanswerable questions have no gold span.
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
"""Generates answers for every question, either through the real agent graph or as a closed-book baseline.

Stops cleanly when the LLM backend is exhausted, and a rerun continues from the first unanswered question.
"""
import argparse
import asyncio
import logging
import time

from services.api.app.agents.graph import agent_app
from services.api.app.agents.state import AgentState
from services.api.app.clients.llm.factory import llm_client
from services.api.app.clients.llm.openai_compatible import ModelExhaustedError

from eval import config as eval_config
from eval import storage

logger = logging.getLogger(__name__)

CLOSED_BOOK_SYSTEM_PROMPT = (
    "Answer the question using only your own knowledge. Do not claim access "
    "to any external documents or corpus. If you genuinely don't know, say "
    "so plainly rather than guessing. Be concise."
)


async def _generate_shipped(question: dict) -> None:
    """Answers one question with the production agent graph and saves it under "shipped"."""
    qid = question["id"]
    if storage.has_keys("generation", eval_config.RESULTS_DIR, qid, ["shipped"]):
        logger.info(f"[generate:{qid}] shipped answer already present — skipping")
        return

    # Same initial state the chat route builds, with no history since each question is independent.
    state = AgentState(
        messages=[],
        current_query=question["question"],
        documents=[],
        plan=[],
        action="",
        corpus_id=eval_config.EVAL_CORPUS_ID,
        tool_used="",
        backend_used="",
        model_used="",
        sources=[],
        is_existence_check=False,
    )
    start = time.monotonic()
    result = await agent_app.ainvoke(state)
    latency_ms = int((time.monotonic() - start) * 1000)

    messages = result.get("messages") or []
    answer = messages[-1].get("content", "") if messages else ""

    entry = {
        "answer": answer,
        "documents": result.get("documents", []),
        "sources": result.get("sources", []),
        "tool_used": result.get("tool_used", ""),
        "backend_used": result.get("backend_used", ""),
        "model_used": result.get("model_used", ""),
        "is_existence_check": bool(result.get("is_existence_check")),
        "latency_ms": latency_ms,
    }
    storage.update("generation", eval_config.RESULTS_DIR, qid, {"shipped": entry})
    logger.info(f"[generate:{qid}] shipped done in {latency_ms}ms tool={entry['tool_used']!r}")


async def _generate_closed_book(question: dict) -> None:
    """Answers one question from the model's own knowledge with no retrieval and saves it under "closed_book"."""
    qid = question["id"]
    if storage.has_keys("generation", eval_config.RESULTS_DIR, qid, ["closed_book"]):
        logger.info(f"[generate:{qid}] closed-book answer already present — skipping")
        return

    start = time.monotonic()
    answer = await llm_client.chat_completion(
        messages=[
            {"role": "system", "content": CLOSED_BOOK_SYSTEM_PROMPT},
            {"role": "user", "content": question["question"]},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    latency_ms = int((time.monotonic() - start) * 1000)
    entry = {
        "answer": answer,
        "backend_used": getattr(llm_client, "last_backend_used", "") or llm_client.__class__.__name__,
        "model_used": getattr(llm_client, "last_model_used", ""),
        "latency_ms": latency_ms,
    }
    storage.update("generation", eval_config.RESULTS_DIR, qid, {"closed_book": entry})
    logger.info(f"[generate:{qid}] closed-book done in {latency_ms}ms")


async def run(closed_book: bool) -> None:
    """Generates answers for all questions, stopping early if the backend runs out of quota.

    Args:
        closed_book: If True, runs the no-retrieval baseline instead of the shipped pipeline.
    """
    data = storage.load(eval_config.DATASET_PATH)
    questions = data["questions"]
    await llm_client.start()
    try:
        for q in questions:
            try:
                if closed_book:
                    await _generate_closed_book(q)
                else:
                    await _generate_shipped(q)
            except ModelExhaustedError as e:
                logger.warning(f"[generate] backend exhausted at {q['id']} — stopping for now: {e}")
                logger.warning("[generate] re-run this exact command later (e.g. after Groq's "
                                "daily reset) to continue from the next unanswered question.")
                break
    finally:
        await llm_client.close()


def main():
    """Standalone entrypoint that mirrors `python -m eval.cli generate`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--closed-book", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run(args.closed_book))


if __name__ == "__main__":
    main()
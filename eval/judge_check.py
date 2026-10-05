"""Smoke test that scores one question with all six metrics before the full judge sweep.

It confirms the judge's JSON output validates against ragas's response models, and warns about any metric that comes back null.
"""
import asyncio
import json
import logging

from eval import config as eval_config
from eval import storage

import argparse
from eval.run_judge import CONTEXT_PRECISION_TOP_K, _CLIENTS, _LLMS, _score, build_metrics

logger = logging.getLogger(__name__)

async def main(backend: str = "gemma") -> None:
    """Scores the first question that has both shipped and closed-book answers and prints the results.

    Args:
        backend: Judge backend name, "gemma" or "cohere".
    """
    data = storage.load(eval_config.DATASET_PATH)
    questions = data["questions"]

    target = None
    for q in questions:
        gen = storage.load(storage.question_path("generation", eval_config.RESULTS_DIR, q["id"]))
        if gen.get("shipped") and gen.get("closed_book"):
            target = q
            break
    if target is None:
        print("No question has both a shipped and a closed-book answer yet — run "
              "`python -m eval.cli generate` and `generate --closed-book` for at "
              "least one question first.")
        return

    qid = target["id"]
    print(f"Judge-check using {qid}: {target['question']!r}\n")

    generation = storage.load(storage.question_path("generation", eval_config.RESULTS_DIR, qid))
    retrieval = storage.load(storage.question_path("retrieval", eval_config.RESULTS_DIR, qid))
    chunks = retrieval.get("hybrid_rerank", [])
    all_texts = [c["text"] for c in chunks]
    top_k_texts = all_texts[:CONTEXT_PRECISION_TOP_K]
    shipped_answer = generation["shipped"]["answer"]
    closed_book_answer = generation["closed_book"]["answer"]
    reference = target.get("reference_answer", "")
    question_text = target["question"]

    metrics = build_metrics(backend)
    client = _CLIENTS[backend]
    await client.start()
    try:
        results = {}
        results["context_precision"] = (
            await _score(metrics["context_precision"], "context_precision",
                         user_input=question_text, response=shipped_answer, retrieved_contexts=top_k_texts)
            if top_k_texts else None
        )
        results["context_recall"] = (
            await _score(metrics["context_recall"], "context_recall",
                         user_input=question_text, retrieved_contexts=all_texts, reference=reference)
            if all_texts else None
        )
        results["faithfulness"] = (
            await _score(metrics["faithfulness"], "faithfulness",
                         user_input=question_text, response=shipped_answer, retrieved_contexts=all_texts)
            if all_texts else None
        )
        results["answer_relevancy"] = await _score(
            metrics["answer_relevancy"], "answer_relevancy", user_input=question_text, response=shipped_answer,
        )
        results["answer_correctness"] = await _score(
            metrics["answer_correctness"], "answer_correctness",
            user_input=question_text, response=shipped_answer, reference=reference,
        )
        results["answer_correctness_closed_book"] = await _score(
            metrics["answer_correctness"], "answer_correctness_closed_book",
            user_input=question_text, response=closed_book_answer, reference=reference,
        )
    finally:
        await client.close()

    print(json.dumps(results, indent=2))
    nulls = [k for k, v in results.items() if v is None]
    if nulls:
        print(f"\nWARNING: {nulls} came back null. Bump this module's logging to DEBUG to see the raw {_CLIENTS} response and the exception from {_LLMS}.agenerate() (usually a pydantic ValidationError, meaning {_CLIENTS}'s json_mode output didn't match the schema ragas's own prompt asked for) before running the full judge sweep.")
    else:
        print("\nAll 6 metrics returned a parseable score — safe to run the full sweep.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-backend", choices=["gemma", "cohere"], default="gemma")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main(args.judge_backend))
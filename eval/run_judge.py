"""Scores saved results with ragas metrics using an LLM judge (Gemma or Cohere) and local embeddings.

Retrieval mode scores context precision and recall, and generation mode scores faithfulness, relevancy and correctness.
Unanswerable questions get a custom refusal check instead, failed metrics are saved as None, and reruns only fill in what is missing.
"""
import argparse
import asyncio
import logging
import math
from pydantic import BaseModel

from eval import _ragas_compat  # noqa: F401 — must load before ragas
from ragas.metrics.collections import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
    ContextRecall,
    Faithfulness,
)

from eval import config as eval_config
from eval import storage

from eval.clients.cohere_client import cohere_client
from eval.clients.gemma_client import gemma_client
from eval.judges.ragas_llm import cohere_instructor_llm, gemma_instructor_llm
from eval.judges.ragas_embeddings import fastembed_ragas_embedding

logger = logging.getLogger(__name__)

_CLIENTS = {"gemma": gemma_client, "cohere": cohere_client}
_LLMS = {"gemma": gemma_instructor_llm, "cohere": cohere_instructor_llm}

# Only the top 3 chunks are scored for context precision to limit judge calls.
CONTEXT_PRECISION_TOP_K = 3

REFUSAL_JUDGE_PROMPT = """You are evaluating whether a candidate answer correctly handles an
UNANSWERABLE question in a closed document corpus.

Question:
{question}

Reference answer:
{reference}

Candidate answer:
{answer}

The question is known to be UNANSWERABLE from the provided document corpus.

Evaluate ONLY whether the candidate answer correctly handles that corpus limitation.

Return judgment=true ONLY if the candidate explicitly acknowledges that the requested
information is NOT available in, contained in, discussed by, or supported by the
provided document corpus.

Return judgment=false if the candidate gives the requested information without such
an acknowledgement.

Output JSON only: {{"correct": true or false}}
"""

class _RefusalVerdict(BaseModel):
    """Structured verdict returned by the refusal judge."""
    correct: bool

async def _judge_refusal(question_text: str, reference: str, answer: str, llm) -> bool | None:
    """Asks the judge whether an answer correctly admits the question can't be answered from the corpus.

    Returns:
        True or False for the verdict, or None if the answer is empty or the call failed.
    """
    if not answer:
        return None
    try:
        result = await llm.agenerate(
            REFUSAL_JUDGE_PROMPT.format(question=question_text, reference=reference, answer=answer),
            _RefusalVerdict,
        )
        return result.correct
    except Exception as e:
        logger.warning(f"[judge] correct_refusal call failed: {e}")
        return None

def build_metrics(backend: str) -> dict:
    """Builds the ragas metric objects wired to the chosen judge backend.

    Args:
        backend: Judge backend name, "gemma" or "cohere".
    """
    llm = _LLMS[backend]
    return {
        "faithfulness": Faithfulness(llm=llm),
        "context_precision": ContextPrecisionWithoutReference(llm=llm),
        "context_recall": ContextRecall(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=fastembed_ragas_embedding, strictness=1),
        "answer_correctness": AnswerCorrectness(llm=llm, embeddings=fastembed_ragas_embedding, weights=[0.75, 0.25]),
    }

def _clean(value) -> float | None:
    """Converts a ragas score to a float, mapping None and NaN to None so averages skip them."""
    if value is None:
        return None
    try:
        return None if math.isnan(value) else float(value)
    except TypeError:
        return None


async def _score(metric, label: str, **kwargs) -> float | None:
    """Runs one ragas metric and returns its score, or None if the call fails.

    Args:
        metric: The ragas metric to run.
        label: Name used in the warning log if the call fails.
        **kwargs: Inputs passed to the metric's ascore().
    """
    try:
        result = await metric.ascore(**kwargs)
        return _clean(result.value)
    except Exception as e:
        logger.warning(f"[judge] {label} call failed: {e}")
        return None


async def _judge_retrieval_one(question: dict, metrics: dict) -> None:
    """Scores context precision and recall for one answerable question and saves them."""
    qid = question["id"]
    if storage.has_keys("judge", eval_config.RESULTS_DIR, qid, ["context_precision", "context_recall"]):
        logger.info(f"[judge:{qid}] retrieval metrics already present — skipping")
        return

    generation = storage.load(storage.question_path("generation", eval_config.RESULTS_DIR, qid))
    retrieval = storage.load(storage.question_path("retrieval", eval_config.RESULTS_DIR, qid))
    chunks = retrieval.get("hybrid_rerank", [])
    shipped_answer = generation.get("shipped", {}).get("answer", "")
    reference = question.get("reference_answer", "")
    question_text = question["question"]

    precision = recall = None
    if question.get("answerable"):
        if chunks and shipped_answer:
            top_k_texts = [c["text"] for c in chunks[:CONTEXT_PRECISION_TOP_K]]
            precision = await _score(
                metrics["context_precision"], "context_precision",
                user_input=question_text, response=shipped_answer, retrieved_contexts=top_k_texts,
            )
        if chunks and reference:
            all_texts = [c["text"] for c in chunks]
            recall = await _score(
                metrics["context_recall"], "context_recall",
                user_input=question_text, retrieved_contexts=all_texts, reference=reference,
            )

    storage.update("judge", eval_config.RESULTS_DIR, qid, {
        "context_precision": precision,
        "context_recall": recall,
    })
    logger.info(f"[judge:{qid}] context_precision={precision} context_recall={recall}")


async def _judge_generation_one(question: dict, metrics: dict, llm) -> None:
    """Scores the generated answers for one question and saves the results.

    Answerable questions get faithfulness, relevancy and correctness, and unanswerable ones get refusal verdicts.
    """
    qid = question["id"]
    keys = ["faithfulness", "answer_relevancy", "answer_correctness", "answer_correctness_closed_book"]
    if not question.get("answerable"):
        keys += ["correct_refusal_shipped", "correct_refusal_closed_book"]
    if storage.has_keys("judge", eval_config.RESULTS_DIR, qid, keys):
        logger.info(f"[judge:{qid}] generation metrics already present — skipping")
        return

    generation = storage.load(storage.question_path("generation", eval_config.RESULTS_DIR, qid))
    retrieval = storage.load(storage.question_path("retrieval", eval_config.RESULTS_DIR, qid))
    chunks = retrieval.get("hybrid_rerank", [])
    all_texts = [c["text"] for c in chunks]
    shipped_answer = generation.get("shipped", {}).get("answer", "")
    closed_book_answer = generation.get("closed_book", {}).get("answer", "")
    reference = question.get("reference_answer", "")
    question_text = question["question"]

    # Reuse scores already saved so only missing metrics cost judge calls.
    done = storage.load(storage.question_path("judge", eval_config.RESULTS_DIR, qid))

    faithfulness = done.get("faithfulness")
    relevancy = done.get("answer_relevancy")
    correctness = done.get("answer_correctness")
    correctness_cb = done.get("answer_correctness_closed_book")
    correct_refusal_shipped = done.get("correct_refusal_shipped")
    correct_refusal_closed_book = done.get("correct_refusal_closed_book")

    if question.get("answerable"):
        if faithfulness is None and shipped_answer and all_texts:
            faithfulness = await _score(
                metrics["faithfulness"], "faithfulness",
                user_input=question_text, response=shipped_answer, retrieved_contexts=all_texts,
            )
        if relevancy is None and shipped_answer:
            relevancy = await _score(
                metrics["answer_relevancy"], "answer_relevancy",
                user_input=question_text, response=shipped_answer,
            )
        if correctness is None and shipped_answer and reference:
            correctness = await _score(
                metrics["answer_correctness"], "answer_correctness",
                user_input=question_text, response=shipped_answer, reference=reference,
            )
        if correctness_cb is None and closed_book_answer and reference:
            correctness_cb = await _score(
                metrics["answer_correctness"], "answer_correctness_closed_book",
                user_input=question_text, response=closed_book_answer, reference=reference,
            )
    else:
        if correct_refusal_shipped is None and shipped_answer:
            correct_refusal_shipped = await _judge_refusal(question_text, reference, shipped_answer, llm)
        if correct_refusal_closed_book is None and closed_book_answer:
            correct_refusal_closed_book = await _judge_refusal(question_text, reference, closed_book_answer, llm)

    storage.update("judge", eval_config.RESULTS_DIR, qid, {
        "faithfulness": faithfulness,
        "answer_relevancy": relevancy,
        "answer_correctness": correctness,
        "answer_correctness_closed_book": correctness_cb,
        "correct_refusal_shipped": correct_refusal_shipped,
        "correct_refusal_closed_book": correct_refusal_closed_book,
    })
    logger.info(
        f"[judge:{qid}] faithfulness={faithfulness} relevancy={relevancy} "
        f"correctness={correctness} correctness_cb={correctness_cb} "
        f"correct_refusal_shipped={correct_refusal_shipped} correct_refusal_closed_book={correct_refusal_closed_book}"
    )


async def run(stage: str, backend: str = "gemma") -> None:
    """Judges every question for one stage.

    Args:
        stage: "retrieval" or "generation".
        backend: Judge backend name, "gemma" or "cohere".
    """
    data = storage.load(eval_config.DATASET_PATH)
    questions = data["questions"]
    metrics = build_metrics(backend)
    client = _CLIENTS[backend]
    await client.start()
    try:
        for q in questions:
            if stage == "retrieval":
                await _judge_retrieval_one(q, metrics)
            else:
                await _judge_generation_one(q, metrics, _LLMS[backend])
    finally:
        await client.close()


def main():
    """Standalone entrypoint that mirrors `python -m eval.cli judge`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", action="store_true")
    parser.add_argument("--generation", action="store_true")
    parser.add_argument("--judge-backend", choices=["gemma", "cohere"], default="gemma")
    args = parser.parse_args()
    if args.retrieval == args.generation:
        parser.error("pass exactly one of --retrieval / --generation")
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run("retrieval" if args.retrieval else "generation", args.judge_backend))


if __name__ == "__main__":
    main()
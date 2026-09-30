"""
Run 5 (--retrieval) and Run 6 (--generation) — the 6 metrics from
EVALUATION_DESIGN.md's table, computed by ragas's REAL metric
implementations (ragas.metrics.collections), pointed at:
  - llm: GemmaInstructorLLM (eval/judges/ragas_llm.py) — Gemma, free
    tier, a different vendor from both the generator (Groq) and the
    captioner (Mistral), same isolation reasoning used elsewhere in this
    project.
  - embeddings: FastEmbedRagasEmbedding (eval/judges/ragas_embeddings.py)
    — the app's own local FastEmbed model, zero API cost, for
    AnswerRelevancy's and AnswerCorrectness's semantic-similarity steps.

Real per-question call count — verified by reading AND running ragas's
actual ascore() implementations (see chat history for the
reproduction), not assumed:

  context_precision   up to 3 calls  ContextPrecisionWithoutReference makes
                       1 call PER context in retrieved_contexts — we
                       deliberately cap that list to the top 3 of
                       hybrid_rerank to hit this budget, rather than
                       accepting ragas's default of scoring all 5.
  context_recall      1 call         ContextRecall makes exactly 1 call
                       regardless of context count (it joins them into
                       one string), so this uses the FULL hybrid_rerank
                       list for a better recall signal.
  faithfulness         up to 2 calls  Faithfulness: extract statements from
                       the answer, then verify each against context — 0
                       if no statements were extracted (e.g. a refusal).
  answer_relevancy    exactly 1 call  AnswerRelevancy(strictness=1) —
                       ragas's own default is strictness=3 (3 synthetic
                       questions); capped to 1 here.
                       + 2 local FastEmbed calls, zero API cost.
  answer_correctness  up to 3 calls  AnswerCorrectness: generate statements
  (shipped)             from the response, generate statements from the
                       reference, classify (skipped if either statement
                       list came back empty) + local embed calls.
  answer_correctness  up to 3 calls  same metric instance, invoked again
  (closed-book)          on the closed-book answer (EVALUATION_DESIGN.md's
                       Baseline row).

  = up to 13 Gemma calls/question x 25 questions = up to 325 total,
  comfortably inside Gemma's 14,400/day free cap. This is ragas's real
  call shape for the knobs we control (context truncation, strictness),
  not a number picked to hit a target — see eval/RUNBOOK.md.
"""
import argparse
import asyncio
import logging
import math
from pydantic import BaseModel

from eval import _ragas_compat  # noqa: F401 — import-order fix, see that module
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

CONTEXT_PRECISION_TOP_K = 3  # budget cap — see module docstring

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
    correct: bool

async def _judge_refusal(question_text: str, reference: str, answer: str, llm) -> bool | None:
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
    llm = _LLMS[backend]
    return {
        "faithfulness": Faithfulness(llm=llm),
        "context_precision": ContextPrecisionWithoutReference(llm=llm),
        "context_recall": ContextRecall(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=fastembed_ragas_embedding, strictness=1),
        "answer_correctness": AnswerCorrectness(llm=llm, embeddings=fastembed_ragas_embedding, weights=[0.75, 0.25]),
    }

def _clean(value) -> float | None:
    """ragas's own "couldn't score this" signal is NaN (e.g. zero
    statements extracted) — convert to None so report.py's
    None-skipping average logic treats it as missing rather than
    propagating NaN through a sum()."""
    if value is None:
        return None
    try:
        return None if math.isnan(value) else float(value)
    except TypeError:
        return None


async def _score(metric, label: str, **kwargs) -> float | None:
    try:
        result = await metric.ascore(**kwargs)
        return _clean(result.value)
    except Exception as e:
        logger.warning(f"[judge] {label} call failed: {e}")
        return None


async def _judge_retrieval_one(question: dict, metrics: dict) -> None:
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

    # Per-metric resume: reuse any score already saved, so only missing
    # metrics cost API calls.
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
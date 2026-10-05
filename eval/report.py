"""Aggregates saved retrieval, generation and judge results into a markdown report.

It makes no API calls, shows missing metrics as n/a instead of 0, and is safe to rerun at any time.
"""
import logging

from eval import config as eval_config
from eval import storage
from eval.metrics import deterministic

logger = logging.getLogger(__name__)

RETRIEVAL_VARIANTS = ("dense_only", "bm25_only", "hybrid", "hybrid_rerank", "rewritten_query")
JUDGE_KEYS = (
    "context_precision", "context_recall", "faithfulness",
    "answer_relevancy", "answer_correctness", "answer_correctness_closed_book",
)


def _avg(values: list) -> float | None:
    """Returns the mean of the non-None values rounded to 3 places, or None if there are none."""
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 3) if clean else None


def _percentile(values: list, pct: float) -> float | None:
    """Returns the value at the given percentile (0 to 1) among non-None values, or None if there are none."""
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return None
    idx = min(len(clean) - 1, int(len(clean) * pct))
    return clean[idx]


def _blend(mean_answerable, n_answerable, refusal_rate, n_unanswerable):
    """Combines the answerable mean score and the refusal rate into one weighted score over all questions."""
    if mean_answerable is None or refusal_rate is None:
        return None
    return round((n_answerable * mean_answerable + n_unanswerable * refusal_rate)
                 / (n_answerable + n_unanswerable), 3)


def build_report() -> dict:
    """Reads every saved result file and returns the aggregated report as a dict."""
    questions = storage.load(eval_config.DATASET_PATH)["questions"]
    n_answerable_q = sum(1 for q in questions if q.get("answerable"))

    retrieval_metrics = {v: {"hit5": [], "mrr": [], "prec5": []} for v in RETRIEVAL_VARIANTS}
    judge_metrics = {k: [] for k in JUDGE_KEYS}  # Answerable questions only.
    latencies_shipped, latencies_cb = [], []
    citation_scores = []
    answered_count = 0
    refusal = {v: {"correct": 0, "correct_n": 0, "false": 0, "n_ans": 0} for v in ("shipped", "closed_book")}
    REFUSAL_JUDGE_KEYS = {"shipped": "correct_refusal_shipped", "closed_book": "correct_refusal_closed_book"}

    for q in questions:
        qid, gold_span, answerable = q["id"], q.get("gold_span"), q.get("answerable")
        retrieval = storage.load(storage.question_path("retrieval", eval_config.RESULTS_DIR, qid))
        generation = storage.load(storage.question_path("generation", eval_config.RESULTS_DIR, qid))
        judge = storage.load(storage.question_path("judge", eval_config.RESULTS_DIR, qid))

        for variant in RETRIEVAL_VARIANTS:
            chunks = retrieval.get(variant, [])
            retrieval_metrics[variant]["hit5"].append(deterministic.hit_at_k(chunks, gold_span, 5))
            retrieval_metrics[variant]["mrr"].append(deterministic.mrr(chunks, gold_span))
            retrieval_metrics[variant]["prec5"].append(deterministic.precision_at_k(chunks, gold_span, 5))

        if answerable:
            for key in JUDGE_KEYS:
                if judge.get(key) is not None:
                    judge_metrics[key].append(judge[key])

        for variant in ("shipped", "closed_book"):
            entry = generation.get(variant)
            if not entry:
                continue
            r = refusal[variant]
            if answerable:
                r["n_ans"] += 1
                r["false"] += deterministic.is_refusal_shaped(entry.get("answer", ""))
            else:
                verdict = judge.get(REFUSAL_JUDGE_KEYS[variant])
                if verdict is not None:  # Unjudged answers are excluded, not counted as wrong.
                    r["correct_n"] += 1
                    r["correct"] += bool(verdict)

        shipped = generation.get("shipped")
        if shipped:
            answered_count += 1
            latencies_shipped.append(shipped.get("latency_ms"))
            citation_scores.append(deterministic.citation_validity(shipped.get("answer", ""), shipped.get("sources", [])))
        if generation.get("closed_book"):
            latencies_cb.append(generation["closed_book"].get("latency_ms"))

    # Per-category results use hybrid_rerank, the shipped pipeline's retrieval method.
    by_category: dict = {}
    for q in questions:
        if not q.get("answerable"):
            continue
        cat = q["category"]
        qid, gold_span = q["id"], q.get("gold_span")
        retrieval = storage.load(storage.question_path("retrieval", eval_config.RESULTS_DIR, qid))
        judge = storage.load(storage.question_path("judge", eval_config.RESULTS_DIR, qid))
        chunks = retrieval.get("hybrid_rerank", [])
        c = by_category.setdefault(cat, {"n": 0, "hit5": [], "mrr": [], "correctness": [], "faithfulness": []})
        c["n"] += 1
        c["hit5"].append(deterministic.hit_at_k(chunks, gold_span, 5))
        c["mrr"].append(deterministic.mrr(chunks, gold_span))
        c["correctness"].append(judge.get("answer_correctness"))
        c["faithfulness"].append(judge.get("faithfulness"))
    by_category = {
        cat: {
            "n": c["n"], "hit@5": _avg(c["hit5"]), "mrr": _avg(c["mrr"]),
            "correctness": _avg(c["correctness"]), "faithfulness": _avg(c["faithfulness"]),
        }
        for cat, c in by_category.items()
    }

    def rate(num, den):
        """Returns num/den rounded to 3 places, or None when den is zero."""
        return round(num / den, 3) if den else None

    abstention, correctness = {}, {}
    for variant, key in (("shipped", "answer_correctness"), ("closed_book", "answer_correctness_closed_book")):
        r = refusal[variant]
        abstention[variant] = {
            "correct_refusal_rate": rate(r["correct"], r["correct_n"]), "n_unanswerable": r["correct_n"], "correct_ans": r["correct"],
            "false_refusal_rate": rate(r["false"], r["n_ans"]), "n_answerable": r["n_ans"], "false_ans": r["false"]
        }
        mean, n = _avg(judge_metrics[key]), len(judge_metrics[key])
        correctness[variant] = {
            "answerable_mean": mean, "n_answerable": n,
            "blended": _blend(mean, n, abstention[variant]["correct_refusal_rate"], r["correct_n"]),
        }

    return {
        "questions_total": len(questions),
        "questions_answerable": n_answerable_q,
        "answered": answered_count,
        "by_category": by_category,
        "judge_metric_values_recorded": sum(len(v) for v in judge_metrics.values()),
        "judge_metric_values_possible": len(JUDGE_KEYS) * n_answerable_q,
        "retrieval_ablation": {
            variant: {"hit@5": _avg(m["hit5"]), "mrr": _avg(m["mrr"]), "precision@5": _avg(m["prec5"])}
            for variant, m in retrieval_metrics.items()
        },
        "judge_metrics": {k: _avg(v) for k, v in judge_metrics.items()},
        "correctness": correctness,
        "abstention": abstention,
        "citation_validity": _avg(citation_scores),
        "latency_ms": {
            "shipped_p50": _percentile(latencies_shipped, 0.5),
            "shipped_p95": _percentile(latencies_shipped, 0.95),
            "closed_book_p50": _percentile(latencies_cb, 0.5),
        },
    }


def render_markdown(report: dict) -> str:
    """Formats the report dict as markdown text."""
    lines = ["# DocRAG — Baseline Evaluation Report", ""]
    lines.append(
        f"Questions: {report['answered']}/{report['questions_total']} answered; judge metrics: "
        f"{report['judge_metric_values_recorded']}/{report['judge_metric_values_possible']} "
        f"values recorded (answerable questions only, n={report['questions_answerable']})."
    )
    lines += ["", "## Retrieval ablation (answerable questions)", "| Variant | Hit@5 | MRR | Precision@5 |", "|---|---|---|---|"]
    for variant, m in report["retrieval_ablation"].items():
        lines.append(f"| {variant} | {m['hit@5']} | {m['mrr']} | {m['precision@5']} |")
    lines += ["", "## Judge-scored metrics (answerable questions)", "| Metric | Score |", "|---|---|"]
    for k, v in report["judge_metrics"].items():
        if k not in ("answer_correctness", "answer_correctness_closed_book"):
            lines.append(f"| {k} | {v} |")
    lines += ("", "## Abstention")
    for variant, label in (("shipped", "Shipped RAG"), ("closed_book", "Closed-book")):
        a = report["abstention"][variant]
        lines.append(f"- Correct Refusal rate ({label}): {a['correct_refusal_rate']} ({a['correct_ans']}/{a['n_unanswerable']})")
    for variant, label in (("shipped", "Shipped RAG"), ("closed_book", "Closed-book")):
        a = report["abstention"][variant]
        lines.append(f"- False-refusal rate ({label}): {a['false_refusal_rate']} ({a['false_ans']}/{a['n_answerable']})")
    lines += ["", "## Answer correctness", "| Answer source | Mean (answerable) | Correct-refusal (unanswerable) | Overall (all) |", "|---|---|---|---|"]
    for variant, label in (("shipped", "Shipped RAG"), ("closed_book", "Closed-book")):
        c, a = report["correctness"][variant], report["abstention"][variant]
        lines.append(
            f"| {label} | {c['answerable_mean']} (n={c['n_answerable']}) | "
            f"{a['correct_refusal_rate']} (n={a['n_unanswerable']}) | {c['blended']} |"
        )
    lines += ["", "## By category (answerable questions)", "| Category | n | Hit@5 | MRR | Correctness | Faithfulness |", "|---|---|---|---|---|---|"]
    for cat, m in report["by_category"].items():
        lines.append(f"| {cat} | {m['n']} | {m['hit@5']} | {m['mrr']} | {m['correctness']} | {m['faithfulness']} |")
    lines += ["", f"## Citation validity: {report['citation_validity']}", "", "## Latency (ms)"]
    lat = report["latency_ms"]
    lines.append(f"- Shipped pipeline: p50={lat['shipped_p50']}, p95={lat['shipped_p95']}")
    lines.append(f"- Closed-book: p50={lat['closed_book_p50']}")
    return "\n".join(lines)


def main() -> None:
    """Builds the report, writes it to results/report.md, and prints it."""
    report = build_report()
    markdown = render_markdown(report)
    out_path = eval_config.RESULTS_DIR / "report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown)
    print(markdown)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
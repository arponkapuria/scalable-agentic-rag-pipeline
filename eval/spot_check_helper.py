"""One-off script that prints six answers (lowest, median and highest correctness) for manual spot-checking against the report."""

import json
from eval import config as eval_config
from eval import storage

# Answerable questions only, keyed by id.
questions = {q["id"]: q for q in storage.load(eval_config.DATASET_PATH)["questions"] if q.get("answerable")}
scored = []
for qid, q in questions.items():
    judge = storage.load(storage.question_path("judge", eval_config.RESULTS_DIR, qid))
    if judge.get("answer_correctness") is not None:
        scored.append((qid, judge["answer_correctness"]))
scored.sort(key=lambda x: x[1])
# Two lowest, two around the median, two highest.
picks = scored[:2] + scored[len(scored)//2-1:len(scored)//2+1] + scored[-2:]

for qid, cohere_score in picks:
    q = questions[qid]
    gen = storage.load(storage.question_path("generation", eval_config.RESULTS_DIR, qid))
    print(f"\n{'='*60}\n{qid} | Cohere score: {cohere_score}")
    print(f"Q: {q['question']}\nReference: {q['reference_answer']}")
    print(f"Shipped answer: {gen['shipped']['answer']}")
# OmniRAG Phase 9a — Baseline Evaluation Report

Questions: 25/25 answered; judge metrics: 126/126 values recorded (answerable questions only, n=21).

## Retrieval ablation (answerable questions)
| Variant | Hit@5 | MRR | Precision@5 |
|---|---|---|---|
| dense_only | 0.571 | 0.421 | 0.238 |
| bm25_only | 0.571 | 0.516 | 0.286 |
| hybrid | 0.619 | 0.532 | 0.276 |
| hybrid_rerank | 0.667 | 0.563 | 0.305 |
| rewritten_query | 0.619 | 0.508 | 0.295 |

## Judge-scored metrics (answerable questions)
| Metric | Score |
|---|---|
| context_precision | 0.865 |
| context_recall | 0.952 |
| faithfulness | 0.705 |
| answer_relevancy | 0.757 |

## Abstention
- Correct Refusal rate (Shipped RAG): 1.0 (4/4)
- Correct Refusal rate (Closed-book): 0.75 (3/4)
- False-refusal rate (Shipped RAG): 0.143 (3/21)
- False-refusal rate (Closed-book): 0.0 (0/21)

## Answer correctness
| Answer source | Mean (answerable) | Correct-refusal (unanswerable) | Overall (all) |
|---|---|---|---|
| Shipped RAG | 0.77 (n=21) | 1.0 (n=4) | 0.807 |
| Closed-book | 0.708 (n=21) | 0.75 (n=4) | 0.715 |

## By category (answerable questions)
| Category | n | Hit@5 | MRR | Correctness | Faithfulness |
|---|---|---|---|---|---|
| single_fact | 7 | 0.857 | 0.69 | 0.833 | 0.56 |
| multi_passage | 5 | 0.8 | 0.7 | 0.801 | 0.971 |
| table_lookup | 4 | 0.5 | 0.375 | 0.592 | 0.646 |
| figure_caption | 2 | 0.5 | 0.5 | 0.818 | 0.688 |
| cross_paper | 3 | 0.333 | 0.333 | 0.775 | 0.689 |

## Citation validity: 1.0

## Latency (ms)
- Shipped pipeline: p50=6041, p95=9909
- Closed-book: p50=1121

## Human spot-check (judge validation)

6 answers selected (2 highest, 2 lowest, 2 median Cohere `answer_correctness` scores) and hand-scored blind before comparing to Cohere's number.

| Question | Cohere score | My score (0-1) | Agree? (within 0.15) | Notes |
|---|---|---|---|---|
| q13 | 0.1363569319475432 | 0.0 | Yes | False refusal — reference confirms 27.3 BLEU is in the paper, model claims no info. Not a correctness issue, a retrieval-recall miss (table content likely missed by chunking/retrieval). |
| q16 | 0.4347561801193094 | 0.0 | No | Same failure mode as q13 — false refusal on a table-lookup fact that exists in the corpus. |
| q17 | 0.8487006693980463 | 1.0 | Yes | Matches reference exactly (encoder-decoder architecture), correctly cited, no extra unsupported claims. |
| q05 | 0.8585306212791092 | 1.0 | Yes | Exact match ("30,000 tokens"), cited. |
| q06 | 0.9396049823911019 | 1.0 | Yes | Exact match ("340M"), cited. |
| q20 | 0.9571455783730805 | 1.0 | Yes | Matches reference content and framing precisely, cited, no hallucinated detail beyond source. |

**How I scored:** 1.0 = fact matches reference, properly attributed, nothing fabricated. 0.0 = no usable answer given (a refusal on an answerable question is a correctness failure, same as a wrong fact — it fails the user either way). I didn't use partial credit (0.5) here because none of these 6 were partially-right-partially-wrong; if we hit one that's half-correct, that's exactly where the 0.15 tolerance band earns its keep.

**Agree rule:** |my_score − cohere_score| ≤ 0.15 → Agree = Yes, else No. 

**Agreement: 5/6 (83%).** Cohere tracks my judgment closely on genuine answers (q05, q06, q17, q20 — all within ~0.15 of a full-credit score), but is too lenient specifically on refusals: it scored q16's flat "I don't have information" refusal at 0.43 instead of 0.0, and q13's came in at 0.14 — inside my agreement threshold, but close enough to flag. Likely cause: `AnswerCorrectness`'s claim-extraction step pulls a weak claim out of refusal boilerplate instead of recognizing zero factual content. Given both disagreements are on refusals specifically, I'd treat `answer_correctness` on shipped answers that turn out to be refusals with some skepticism, and lean on the deterministic false-refusal rate (which already correctly flags these as failures) as the more reliable signal for that specific failure mode.
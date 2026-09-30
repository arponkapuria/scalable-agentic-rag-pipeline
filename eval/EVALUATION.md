# Evaluation - Design & Runbook

This documents the baseline evaluation I ran on the shipped pipeline: what it measures and why, how it was set up, and how to reproduce it. Every stage runs through the production code paths directly, nothing is reimplemented for eval. Results are in `eval/report.md`.

Every command is resumable. If a run stops (daily rate limit, crash, closed terminal), running the same command again continues from the first unfinished question. Finished work is not redone and no API call is paid for twice.

All commands were run from the repo root, on the `trim-rebuild` branch.

---

## Design

### Corpus and questions

Five papers: Transformer, ResNet, BERT, LoRA, RAG (Lewis et al.). 25 questions, each with a human-verified reference answer and gold chunk span, spread across the paths a real literature-survey session would hit.

| Category | Count | Why |
|---|---|---|
| Single-fact | 7 | Most common query shape, easiest to verify by hand |
| Multi-passage | 5 | Does retrieval return every needed chunk, not just one |
| Table lookup | 4 | Covers Docling's table serialization directly |
| Figure/caption | 2 | Few exist in the corpus |
| Cross-paper | 3 | Does retrieval scope correctly across the whole corpus |
| Unanswerable | 4 | Needed to measure refusal in both directions |

That is 21 answerable and 4 unanswerable. The 4 unanswerable questions have no gold span by design.

### Metrics

"Scored by": **Deterministic** is plain code, **Judge** is an LLM, **Human** is a manual score.

| Stage | Metric | What it measures | Scored by | On |
|---|---|---|---|---|
| Retrieval | Hit@5 | Gold chunk in the top 5 (exact match on gold span) | Deterministic | 21 answerable |
| Retrieval | MRR | 1/rank of the gold chunk, 0 if absent | Deterministic | 21 answerable |
| Retrieval | Precision@5 | Share of the top 5 that are gold-relevant | Deterministic | 21 answerable |
| Retrieval | Context Precision | Ranking-aware relevance of retrieved chunks, judged per chunk | Judge | 21 answerable |
| Retrieval | Context Recall | Whether the context covers what the reference answer needs | Judge | 21 answerable |
| Ablation | Hit@5 / MRR / Precision@5 | The same three metrics for dense-only, BM25-only, hybrid, hybrid+rerank, and rewritten-query+rerank (the planner refines the query first) | Deterministic | 21 answerable |
| Generation | Faithfulness | Are the answer's claims grounded in the retrieved context | Judge | 21 answerable |
| Generation | Answer Relevancy | Does the answer address the question (regenerates a question from the answer, compares embeddings) | Judge + local embedder | 21 answerable |
| Generation | Answer Correctness | Does the answer match the reference | Judge + local embedder | 21 answerable |
| Baseline | Answer Correctness, closed-book | Same generator with no retrieval, isolating what retrieval adds | Judge + local embedder | 21 answerable |
| Abstention | Correct-refusal rate (shipped + closed-book) | Genuinely declines instead of fabricating (binary) | Judge | 4 unanswerable |
| Abstention | False-refusal rate (shipped + closed-book) | Wrongly refuses an answerable question | Deterministic | 21 answerable |
| End-to-end | Latency p50/p95 | Per-stage wall clock from the generation timestamps | Deterministic | all 25 |
| End-to-end | Citation validity | Cited sources are among the retrieved chunks | Deterministic | all 25 |
| Sanity | Judge vs human | 6 answers hand-scored and compared to the judge | Human | 6 sampled |

The local embedder is the app's own FastEmbed model (bge-large), so it costs nothing.

### Models and rate limits

- **Judge:** Gemma 4 31B (free, 14,400 req/day) or Cohere R-7B-12-2024 (free, 20 rpm, 1,000/month), selected with `--judge-backend gemma|cohere`.
- **Captioning:** Mistral, a different vendor from the judge so the captioner can't bias the scores.
- **Generation:** Groq.

The shared rate limiter skips a call once RPM is hit instead of waiting. That works with a fallback model but nulls every remaining question with a single-model judge, so each judge call sleeps a fixed interval first (2.1s Gemma, 3.1s Cohere).

Free-tier limits drift, so the values hardcoded in `eval/clients/gemma_client.py` are worth rechecking against the live console.

The overall timeline is set by Groq's 200K tokens/day, not wall clock. Judging takes minutes.

### Result files

One file per question per stage, about 75 in total. Each write is an atomic rewrite of one file.

```
results/retrieval/q07.json    {"dense_only": [...], "hybrid_rerank": [...], ...}
results/generation/q07.json   {"shipped": {...}, "closed_book": null}
results/judge/q07.json        {"context_precision": null, "faithfulness": 0.8, ...}
```

---

## Setup

### Docker

Three profiles are enough. There is no `cache` (Redis), because eval skips the chat cache and session layer and talks to the agent graph directly. The `api` container isn't needed either, since the eval scripts import the app package instead of calling the HTTP API.

```bash
make up PROFILE=core,vector,storage
```

`docker compose ps` shows `postgres`, `qdrant` and `minio` as `Up`.

### Environment

Added to `.env`. The existing `GROQ_API_KEY`, `MISTRAL_API_KEY`, `S3_*`, `DATABASE_URL` and `QDRANT_*` are reused as-is. Only the judge backend in use needs a real key.

```
# Gemma judge (key from https://aistudio.google.com/apikey)
GOOGLE_API_KEY=<Google AI Studio key>
GEMMA_JUDGE_MODEL=gemma-4-31b-it
GEMMA_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai

# Cohere judge (key from https://docs.cohere.com/docs/compatibility-api)
COHERE_API_KEY=<Cohere trial key>
COHERE_JUDGE_MODEL=command-r7b-12-2024
COHERE_BASE_URL=https://api.cohere.ai/compatibility/v1
```

### Python

The eval scripts import the app package directly, so the `api` and `ingestion` groups are installed along with `eval`:

```bash
uv sync --group api --group ingestion --group eval
```

`eval` adds `ragas` for the judge metrics. It's the one heavy group: about 30 extra packages (`langchain`, `langchain-community`, `datasets`, `pyarrow`, `instructor`, `openai`, `tiktoken` and others), several of them the same ones removed in the Neo4j/Ray cleanup. It sits in its own dependency group so a plain `uv sync` and the deployed `api` service stay lean.

`ragas==0.4.3` has an upstream bug: `ragas/llms/base.py` imports `ChatVertexAI` from `langchain_community.chat_models.vertexai`, which current `langchain-community` no longer ships, so `import ragas` fails with `ModuleNotFoundError`. `eval/_ragas_compat.py` stubs that import (VertexAI is unused here), and every eval module imports it before ragas. It has to stay.

### Golden set check

Every `reference_answer` and `gold_span` in `eval/dataset/questions.json` was verified by hand against the 5 papers before any run. A wrong `gold_span` silently zeroes Hit@5, MRR and Precision@5 for that question.

---

## Steps

Retrieval, both generation runs and judging are independent commands. Each lists what it needs, and any order that respects those dependencies works.

### 1. Ingest

```bash
python -m eval.cli ingest
```

Downloads the 5 pinned arXiv PDFs, uploads them to MinIO and pushes them through the production ingestion path: Docling parse, chunk, embed, index, then Mistral captions the figures.

- **Calls:** about 80-150 Mistral vision calls (one per non-decorative figure, roughly 15-30 per paper), spaced `CAPTION_MIN_INTERVAL_SECONDS` apart. Nothing to Groq or Gemma.
- **Time:** 5-15 minutes, mostly Docling's CPU parsing (seconds to ~30s per paper on an M1) plus the paced captioning. A fresh environment also downloads about 1GB of Docling weights, once.
- **Result:** `[ingest-corpus] {filename} complete` is logged 5 times, and Postgres has 5 `Document` rows with `corpus_id=eval-baseline-v1`, `status=complete`. `GET /api/v1/corpus/documents` shows the same when the API is up.
- **On interruption:** completed papers are skipped via the `Document` table.

### 2. Verify gold spans

```bash
python -m eval.cli verify-gold
```

Scrolls every indexed chunk for `eval-baseline-v1` out of Qdrant and checks each `gold_span` is a substring of at least one. No calls, a few seconds.

Output on success: `OK: all 21 gold spans found across N indexed chunks.` It's 21 because the 4 unanswerable questions have no gold span.

On failure it lists the missing question IDs. The fix is to find the real wording in the paper or the indexed chunk (via Qdrant's REST API, or the debug dump of the last-ingested paper, since the dump is overwritten per paper) and set `gold_span` to a verbatim substring. Re-checking is free.

Step 3 doesn't start until this passes.

### 3. Retrieval ablation

*Needs step 2.*

```bash
python -m eval.cli retrieve
```

Runs dense-only, BM25-only, hybrid RRF and hybrid+rerank for each of the 25 questions and saves the ranked chunks.

- **Calls:** 0. Qdrant, FastEmbed and the reranker all run locally.
- **Time:** 1-3 minutes, mostly the reranker's ONNX inference.
- **Result:** `eval/results/retrieval/q01.json` to `q25.json`, each with `dense_only`, `bm25_only`, `hybrid` and `hybrid_rerank` keys holding 5-10 chunks. The log ends with `[retrieve] ablation complete for 25 questions.`

### 4. Generate, shipped pipeline

*Needs step 3.*

```bash
python -m eval.cli generate
```

Runs the real LangGraph agent (planner, retriever/tool, responder) as production chat does, single-turn and uncached.

- **Calls:** 1-2 Groq calls per question, about 25-45 total. Existence-check and unanswerable questions may skip the responder through the deterministic zero-hit guard.
- **Time:** seconds per question. The limit is Groq's 200K tokens/day. At 3-5K tokens per RAG call this probably fits in one day, but on `ModelExhaustedError` it logs a warning and stops cleanly.
- **On interruption:** it resumes after Groq's per-model daily reset (~24h), starting at the first question with no `shipped` key.
- **Result:** each `generation/qNN.json` gains a `shipped` key with `answer`, `documents`, `sources`, `tool_used`, `backend_used`, `model_used` and `latency_ms`. For q22-q25 the answers are expected to be refusals or "No, that isn't mentioned..." style, never fabricated.

### 5. Generate, closed-book baseline

```bash
python -m eval.cli generate --closed-book
```

Runs the same 25 questions through a direct `llm_client` call with no retrieval and no corpus, which isolates what RAG adds over the model's own knowledge.

- **Calls:** 25 Groq calls, with much shorter prompts than the shipped run.
- **Time:** a minute or two.
- **Result:** each `generation/qNN.json` gains a `closed_book` key.

For q22-q25 the model may well answer from training data (q24 on Stable Diffusion, for example). That is expected, since only the shipped pipeline is scoped to "not in this corpus."

### 6. Judge pre-flight (checkpoint)

*Needs steps 4 and 5.*

```bash
python -m eval.cli judge-check --judge-backend cohere
```

Takes the first question with both a shipped and a closed-book answer, runs all 6 ragas metrics on it once and prints the raw scores. Up to 13 judge calls (4 retrieval + 9 generation), a few seconds. `eval/run_judge.py`'s docstring has the per-metric breakdown.

Output on success: a JSON block with 6 numeric scores, then `All 6 metrics returned a parseable score — safe to run the full sweep.`

A `null` value comes with a printed warning. Usually the chunk list is empty (step 3 didn't run for that question) or the judge wrapper raised a `pydantic.ValidationError` because the judge's `json_mode` output didn't match ragas's schema. DEBUG logging shows the raw response.

Steps 7 and 8 wait on this passing clean, so the full budget isn't spent on a broken setup.

### 7. Judge, retrieval metrics

```bash
python -m eval.cli judge --retrieval --judge-backend cohere
```

- `context_precision` (`ContextPrecisionWithoutReference`): 1 call per context, capped at the top 3 of `hybrid_rerank`.
- `context_recall` (`ContextRecall`): exactly 1 call.

Up to 4 calls per question, 100 total, a few minutes. `judge/qNN.json` gains `context_precision` and `context_recall`. The 4 unanswerable questions mostly come back `null`, since their retrieved chunks are empty or irrelevant. That's not a bug.

### 8. Judge, generation metrics

```bash
python -m eval.cli judge --generation --judge-backend cohere
```

- `faithfulness`: up to 2 calls (extract statements, then verify).
- `answer_relevancy`: 1 call (`strictness=1`).
- `answer_correctness`: up to 3 calls for the shipped answer and up to 3 more for the closed-book one.
- Refusal judge: 1 binary call each for the shipped and closed-book answers on the 4 unanswerable questions.

Up to 9 calls per question, 225 total, a few minutes. `judge/qNN.json` gains `faithfulness`, `answer_relevancy`, `answer_correctness` and `answer_correctness_closed_book`.

Together with step 7 that's at most 325 judge calls. This is ragas's real call shape under the knobs I control (top-3 context truncation, `strictness=1`), well inside Gemma's 14,400/day. On Cohere's 1,000/month it uses about a third of the month.

### 9. Human spot-check (checkpoint)

`eval/spot_check_helper.py` picks 6 answers from `eval/results/judge/*.json`: the 2 highest, 2 lowest and 2 median-scoring. For each, I read the question, shipped answer and reference answer, scored it by hand, and compared that with the judge's score. No judge number is trusted before this is done.

### 10. Report

```bash
python -m eval.cli report
```

No calls, instant. Writes `eval/report.md` and prints it: the retrieval ablation table (Hit@5, MRR, Precision@5 across all 5 variants), judge-metric averages, abstention rates, citation validity and latency p50/p95.

The "answered X/25" line is the first thing checked. If it's short, an earlier step didn't finish, and the averages can't be trusted until it's re-run. After that, the 2-3 worst-scoring answers get read to understand why they scored low before any number is quoted.

---

## Design choices and reproducibility

**Design choices**

- **Real `ragas` metrics, not hand-rolled prompts.** Every prompt and scoring formula is ragas's own, and only the LLM and embedding backends are custom-wired.
- **Resumable by construction.** Each (stage, question) pair is one JSON file and a call is skipped if its key is already filled, so a crash loses at most one call and any step can be re-run on its own.
- **Correct-refusal is LLM-judged, false-refusal is regex.** Telling a genuine decline from a confident fabrication needs judgment, while telling "refused" from "answered" is pattern matching.
- **`answer_correctness` is reported three ways.** The mean over the 21 answerable, the correct-refusal rate over the 4 unanswerable, and a blended average across all 25: `(21·mean + 4·rate)/25`.
- **Gold chunks are matched by substring, not chunk id.** Qdrant point ids are random per ingestion, so `gold_span` is checked for containment in the retrieved chunk text.

**Reproducibility**

- **Fixed corpus.** `EVAL_CORPUS_ID = "eval-baseline-v1"` and the 5 pinned arXiv URLs (`PAPERS` in `eval/config.py`). arXiv URLs are stable per version, so a re-ingest months later gets the same PDFs.
- **Determinism.** Planner and judge calls use `temperature=0.0`. The responder keeps the production default of 0.3, so re-running `generate` gives similar answers, not identical ones. That's deliberate: the eval measures the shipped pipeline's real behavior.
- **Full reset.** Delete `eval/results/*`, then delete the 5 `Document` rows and their Qdrant points for `corpus_id=eval-baseline-v1`, and start again from step 1. There's no cascade-delete helper for this corpus (it was never a real session, so `session/cleanup.py` doesn't apply), so the delete is a Qdrant filter-delete plus `DELETE FROM documents WHERE corpus_id='eval-baseline-v1'` in Postgres.
- **Where the numbers live.** Raw per-question JSON in `eval/results/` and the aggregate in `eval/report.md`, all plain files that can be committed alongside the code.

## Known limitations

- **n=25 (21 answerable).** A methodology baseline, not a statistically powered result, since one question flipping moves a rate by about 5 points.
- **One judge backend per run,** with no cross-judge agreement beyond the 6-answer human spot check.
- **`gold_span` is a single unscoped string,** which affects only the deterministic retrieval metrics, not the judged ones.
- **The closed-book baseline is contaminated,** since these famous papers are likely memorized, so the retrieval ablation is the clean evidence that RAG helps.
- **`RERANK_TOP_N` (5) was not tested against a higher value,** so whether retrieval misses sit just outside the top 5 is unverified.

## Call budget

| Step | Vendor | Calls | Note |
|---|---|---|---|
| Ingest | Mistral | ~80-150 | Figure captioning, paced |
| Generate, shipped | Groq | ~25-45 | Bound by 200K tokens/day |
| Generate, closed-book | Groq | 25 | Short prompts |
| Judge, retrieval | Gemma or Cohere | up to 100 | Precision (1 per context, top 3) + recall (1) |
| Judge, generation | Gemma or Cohere | up to 225 | Faithfulness + relevancy + correctness x2 |
| **Judge total** | | **up to 325** | |
| Retrieval ablation, report | none | 0 | Local only |

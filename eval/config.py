"""
Eval-specific constants — deliberately NOT added to services/api/app/config.py
(the app's own Settings/.env): these aren't runtime app tunables, they're
fixed properties of ONE offline evaluation run (which corpus, which 5
papers, where results land). GOOGLE_API_KEY/GEMMA_JUDGE_MODEL/GEMMA_BASE_URL
ARE in Settings (config.py) since those are real credentials/judge-model
choice, the same category as GROQ_API_KEY/MISTRAL_API_KEY.
"""
from pathlib import Path

# Fixed, not a real session corpus_id — never created via /session/init, so
# it's never added to Redis's sessions:active sorted set and the session
# TTL cleanup task (session/cleanup.py) can never touch/delete it. Safe to
# leave indexed indefinitely across eval re-runs.
EVAL_CORPUS_ID = "eval-baseline-v1"

EVAL_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_ROOT / "results"
DATASET_PATH = EVAL_ROOT / "dataset" / "questions.json"

# 5 papers, fixed arXiv PDFs — matches EVALUATION_DESIGN.md's corpus.
# Pinned by arXiv id (not "latest version") for reproducibility — arXiv
# papers can get revised; re-fetching these same URLs months from now
# should still return the same PDF bytes.
PAPERS = [
    {
        "name": "transformer",
        "filename": "attention_is_all_you_need.pdf",
        "url": "https://arxiv.org/pdf/1706.03762",
    },
    {
        "name": "resnet",
        "filename": "deep_residual_learning.pdf",
        "url": "https://arxiv.org/pdf/1512.03385",
    },
    {
        "name": "bert",
        "filename": "bert.pdf",
        "url": "https://arxiv.org/pdf/1810.04805",
    },
    {
        "name": "lora",
        "filename": "lora.pdf",
        "url": "https://arxiv.org/pdf/2106.09685",
    },
    {
        "name": "rag",
        "filename": "rag_lewis_et_al.pdf",
        "url": "https://arxiv.org/pdf/2005.11401",
    },
]

# Candidates fetched per retrieval variant, pre-rerank. Reranking then
# trims to settings.RERANK_TOP_N (5) for the hybrid_rerank variant, same
# as production retrieval.
RETRIEVAL_ABLATION_LIMIT = 10
"""Fixed constants for the offline evaluation run."""

from pathlib import Path

# Fixed id that never goes through /session/init, so session cleanup can never delete this corpus.
EVAL_CORPUS_ID = "eval-baseline-v1"

EVAL_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_ROOT / "results"
DATASET_PATH = EVAL_ROOT / "dataset" / "questions.json"

# Five arXiv papers pinned by id so re-fetching always returns the same PDF.
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

# Candidates fetched per retrieval variant before reranking trims to RERANK_TOP_N.
RETRIEVAL_ABLATION_LIMIT = 10
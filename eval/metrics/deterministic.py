"""
Deterministic (no-LLM, zero-cost) metrics — computed directly from saved
retrieval/generation files at report time, per EVALUATION_DESIGN.md's
"Scored by: Deterministic" rows. Gold-chunk matching is substring
containment of the question's `gold_span` (a short, human-verified phrase
expected verbatim in the true chunk) against a retrieved chunk's text,
normalized for whitespace/case — there's no stable chunk id stored in
Qdrant (ingestion's payload is text/filename/page/section/corpus_id only),
so this is the practical alternative that needs no ingestion-side schema
change.
"""
import re


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _is_gold_hit(chunk_text: str, gold_span: str) -> bool:
    if not gold_span:
        return False
    return normalize_text(gold_span) in normalize_text(chunk_text)


def hit_at_k(chunks: list[dict], gold_span: str, k: int) -> float | None:
    """None (not 0) for unanswerable questions — there IS no gold chunk,
    so "did we hit it" is a category error, not a failed retrieval."""
    if not gold_span:
        return None
    return 1.0 if any(_is_gold_hit(c["text"], gold_span) for c in chunks[:k]) else 0.0


def mrr(chunks: list[dict], gold_span: str) -> float | None:
    if not gold_span:
        return None
    for i, c in enumerate(chunks, start=1):
        if _is_gold_hit(c["text"], gold_span):
            return 1.0 / i
    return 0.0


def precision_at_k(chunks: list[dict], gold_span: str, k: int) -> float | None:
    """Fraction of the top-k retrieved chunks that ARE the gold chunk.
    With a single gold span per question (not a full labeled relevant
    set), this collapses to hit@k / k — the honest number for a
    single-gold-chunk dataset, rather than inventing a broader relevance
    judgment this metric wasn't designed to need (that's what Context
    Precision, the Gemma-judged metric, is for)."""
    if not gold_span:
        return None
    top_k = chunks[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for c in top_k if _is_gold_hit(c["text"], gold_span))
    return hits / len(top_k)


_SUBJ = r"(?:paper|papers|authors?|documents?|corpus|study|article|text)"

REFUSAL_PATTERNS = (
    r"\bi (?:do not|don'?t) have (?:that |any |enough |reliable |specific )?information\b",
    r"\bi (?:do not|don'?t) know\b",
    r"\bi(?:'m| am) not aware\b",
    r"\bi (?:cannot|can'?t) (?:answer|provide)\b",
    r"\b(?:not|isn'?t|aren'?t) (?:in|mentioned in|covered (?:by|in)|contained in) (?:the )?(?:provided )?documents\b",
    r"\b(?:the )?(?:provided )?documents (?:do not|don'?t|does not|doesn'?t) (?:contain|discuss|report|mention|cover)\b",
    rf"\b{_SUBJ} (?:does|did|do) not (?:report|present|discuss|include|contain|cover|mention)\b",
    rf"\b{_SUBJ} (?:doesn'?t|didn'?t|don'?t) (?:report|present|discuss|include|contain|cover|mention)\b",
)

_COMPILED_REFUSAL_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in REFUSAL_PATTERNS)

def is_refusal_shaped(answer: str) -> bool:
    normalized = normalize_text(answer).replace("\u2019", "'")
    return any(p.search(normalized) for p in _COMPILED_REFUSAL_PATTERNS)


def citation_validity(answer: str, sources: list[str]) -> float | None:
    """Fraction of '[Source: X]' citations in the answer whose X is
    actually among the sources this generation run retrieved. None (not
    0/1) when the answer cites nothing at all — an uncited answer isn't a
    citation-validity failure, it's a different question (often a
    refusal, where citing nothing is correct)."""
    cited = re.findall(r"\[Source:\s*([^\]]+)\]", answer or "")
    if not cited:
        return None
    valid = sum(1 for c in cited if c.strip() in (sources or []))
    return valid / len(cited)
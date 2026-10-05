"""Deterministic metrics computed without any LLM calls.

Retrieval metrics match each question's gold span against chunk text, and generation metrics detect refusals and check citations.
"""
import re


def normalize_text(text: str) -> str:
    """Lowercases text and collapses all whitespace to single spaces."""
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _is_gold_hit(chunk_text: str, gold_span: str) -> bool:
    """Returns True if the gold span appears in the chunk after normalization."""
    if not gold_span:
        return False
    return normalize_text(gold_span) in normalize_text(chunk_text)


def hit_at_k(chunks: list[dict], gold_span: str, k: int) -> float | None:
    """Returns 1.0 if the gold chunk is in the top k, else 0.0.

    Args:
        chunks: Retrieved chunks in rank order, each with a "text" key.
        gold_span: Expected phrase, empty for unanswerable questions.
        k: Number of top chunks to check.

    Returns:
        The score, or None for unanswerable questions since there is no gold chunk.
    """
    if not gold_span:
        return None
    return 1.0 if any(_is_gold_hit(c["text"], gold_span) for c in chunks[:k]) else 0.0


def mrr(chunks: list[dict], gold_span: str) -> float | None:
    """Returns the reciprocal rank of the first gold chunk, 0.0 if absent, or None if there is no gold span."""
    if not gold_span:
        return None
    for i, c in enumerate(chunks, start=1):
        if _is_gold_hit(c["text"], gold_span):
            return 1.0 / i
    return 0.0


def precision_at_k(chunks: list[dict], gold_span: str, k: int) -> float | None:
    """Returns the fraction of the top k chunks containing the gold span, or None if there is no gold span."""
    if not gold_span:
        return None
    top_k = chunks[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for c in top_k if _is_gold_hit(c["text"], gold_span))
    return hits / len(top_k)


_SUBJ = r"(?:paper|papers|authors?|documents?|corpus|study|article|text)"

# Phrases that signal the answer is declining to answer.
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
    """Returns True if the answer reads like a refusal or an admission of not knowing."""
    normalized = normalize_text(answer).replace("\u2019", "'")
    return any(p.search(normalized) for p in _COMPILED_REFUSAL_PATTERNS)


def citation_validity(answer: str, sources: list[str]) -> float | None:
    """Returns the fraction of "[Source: X]" citations that match a retrieved source.

    Args:
        answer: The generated answer text.
        sources: Source names retrieved for this answer.

    Returns:
        The fraction, or None if the answer has no citations.
    """
    cited = re.findall(r"\[Source:\s*([^\]]+)\]", answer or "")
    if not cited:
        return None
    valid = sum(1 for c in cited if c.strip() in (sources or []))
    return valid / len(cited)
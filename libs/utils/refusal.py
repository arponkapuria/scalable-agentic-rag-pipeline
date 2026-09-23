"""
Shared answer-quality markers: refusal detection used by both chat.py
(cache-gating) and responder.py (post-generation offer-to-search), plus the
truncation marker responder.py appends and chat.py must recognize (never
cache a cut-off answer). Lives here, not in
chat.py, specifically so responder.py can import it without a circular
import (routes/chat.py -> agents/graph.py -> agents/nodes/responder.py,
so responder.py importing anything from routes/chat.py would cycle back).
"""

# Guardrail-blocked messages (libs/guardrails) live here, not in that module,
# for the same circular-import reason as everything else in this file: the
# input-guard path in routes/chat.py and the output-guard path in
# agents/nodes/responder.py both need one shared constant, and this module
# already sits below both without cycling back to either.
GUARDRAIL_INPUT_BLOCKED = "I can't help with that request."
GUARDRAIL_OUTPUT_BLOCKED = "I can't share that response."

REFUSAL_PREFIXES = (
    "I don't have information about",
    "I don't have that information in my documents.",
    GUARDRAIL_OUTPUT_BLOCKED,  # never cache a guardrail-blocked answer
)


def is_refusal(answer: str) -> bool:
    return any(answer.startswith(p) for p in REFUSAL_PREFIXES)


# Appended when the model stopped at the token cap (finish_reason == "length").
# Without it a cut-off answer is indistinguishable from a finished one.
TRUNCATION_NOTE = "\n\n*(Answer cut off — token limit reached.)*"


def is_truncated(answer: str) -> bool:
    return answer.endswith(TRUNCATION_NOTE)

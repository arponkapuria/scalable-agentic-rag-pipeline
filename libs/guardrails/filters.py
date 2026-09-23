"""
Two check functions (input, output) plus guard_output(), the single call
site that applies the output check to whatever generate_node produced —
so responder.py's two answer-generating paths (_answer_from_documents,
_answer_conversationally) and the sandbox echo path all get covered by
ONE wrapper instead of three separate checks.
"""
import logging
from typing import Optional

from libs.guardrails.patterns import INPUT_PATTERNS, MAX_INPUT_CHARS, OUTPUT_PATTERNS
from libs.utils.refusal import GUARDRAIL_OUTPUT_BLOCKED

logger = logging.getLogger(__name__)


def check_input(text: str) -> Optional[str]:
    """Returns the label of the first pattern that matched, or None."""
    if len(text) > MAX_INPUT_CHARS:
        return "input_too_long"
    for pattern, label in INPUT_PATTERNS:
        if pattern.search(text):
            return label
    return None


def check_output(text: str) -> Optional[str]:
    for pattern, label in OUTPUT_PATTERNS:
        if pattern.search(text):
            return label
    return None


def guard_output(result: dict) -> dict:
    """Applied once, in generate_node, to whichever branch produced the
    final answer — mutates result['messages'][-1] in place if the output
    guard fires, and returns the same dict either way. backend_used/
    model_used are cleared to match the shape every other system-produced
    (non-model) message already uses in responder.py, so a blocked answer
    reads identically to a deterministic refusal downstream."""
    messages = result.get("messages") or []
    if not messages:
        return result
    answer = messages[-1].get("content", "")
    reason = check_output(answer)
    if reason:
        logger.info(f"Output guardrail blocked a response ({reason})")
        messages[-1]["content"] = GUARDRAIL_OUTPUT_BLOCKED
        result["backend_used"] = "none"
        result["model_used"] = ""
    return result

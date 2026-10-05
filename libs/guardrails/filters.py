"""
Guardrail check functions. check_input validates a user message before it reaches the LLM, check_output validates a generated answer before it's returned, and guard_output applies the output check in one place for every answer-generating code path in the agent.
"""
import logging
from typing import Optional

from libs.guardrails.patterns import INPUT_PATTERNS, MAX_INPUT_CHARS, OUTPUT_PATTERNS
from libs.utils.refusal import GUARDRAIL_OUTPUT_BLOCKED

logger = logging.getLogger(__name__)


def check_input(text: str) -> Optional[str]:
    """Checks a user message against the length cap and injection/jailbreak patterns.

    Args:
        text: Raw user message.

    Returns:
        The label of the first pattern that matched (or "input_too_long"), or None if the input is clean.
    """
    if len(text) > MAX_INPUT_CHARS:
        return "input_too_long"
    for pattern, label in INPUT_PATTERNS:
        if pattern.search(text):
            return label
    return None


def check_output(text: str) -> Optional[str]:
    """Checks a generated answer for system-prompt leakage.

    Args:
        text: The model's generated answer text.

    Returns:
        The label of the first pattern that matched, or None if clean.
    """
    for pattern, label in OUTPUT_PATTERNS:
        if pattern.search(text):
            return label
    return None


def guard_output(result: dict) -> dict:
    """Runs check_output on the agent's final answer and replaces it with a fixed message if it fails.

    Args:
        result: The agent state dict, containing at least "messages".

    Returns:
        The same dict, with the last message replaced and backend_used/model_used cleared if the guard fired.
    """
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
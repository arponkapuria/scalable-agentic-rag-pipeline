"""
Shared constants and helper functions for detecting refusal answers and truncated answers, so the same detection logic is used both when writing to the cache and when reading from it.
"""

GUARDRAIL_INPUT_BLOCKED = "I can't help with that request."
GUARDRAIL_OUTPUT_BLOCKED = "I can't share that response."

REFUSAL_PREFIXES = (
    "I don't have information about",
    "I don't have that information in my documents.",
    GUARDRAIL_OUTPUT_BLOCKED,  # never cache a guardrail-blocked answer
)


def is_refusal(answer: str) -> bool:
    """Checks whether an answer is a refusal or non-answer that should never be cached.

    Args:
        answer: The final answer text.

    Returns:
        True if the answer starts with any REFUSAL_PREFIXES entry.
    """
    return any(answer.startswith(p) for p in REFUSAL_PREFIXES)


# Appended when the model stops at the token cap, so a cut-off answer is distinguishable from a finished one.
TRUNCATION_NOTE = "\n\n*(Answer cut off — token limit reached.)*"


def is_truncated(answer: str) -> bool:
    """Checks whether an answer was cut off at the token cap.

    Args:
        answer: The final answer text.

    Returns:
        True if the answer ends with TRUNCATION_NOTE.
    """
    return answer.endswith(TRUNCATION_NOTE)
"""
Validates and normalizes the planner's parsed JSON output, replacing an invalid or missing field with a safe default instead of raising.
"""
from typing import Any

_VALID_ACTIONS = {"retrieve", "direct_answer", "tool_use"}
_VALID_TOOL_CHOICES = {"web_search", "sandbox", None}


def validate_plan(plan: Any) -> dict:
    """Coerces an invalid action or tool_choice to a safe default.

    Args:
        plan: The planner LLM's parsed JSON output (or anything else, if parsing failed upstream).

    Returns:
        A dict with valid "action" and "tool_choice" values. Empty dict if `plan` wasn't a dict to begin with.
    """
    if not isinstance(plan, dict):
        return {}
    validated = dict(plan)
    if validated.get("action") not in _VALID_ACTIONS:
        validated["action"] = "retrieve"
    if validated.get("tool_choice") not in _VALID_TOOL_CHOICES:
        validated["tool_choice"] = None
    return validated
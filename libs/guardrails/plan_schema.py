"""
Validates the planner's parsed JSON against the values planner.py's own
downstream code actually expects. A safety net on top of the existing
`.get(...) or default` fallback (see planner.py's docstring on the
Phase 6 null-handling fix) — NEVER stricter than that fallback, since a
validator that rejects a case the lenient code used to handle would be a
regression, not an improvement. An invalid/out-of-domain field is
replaced with the same default the old code already fell back to;
nothing here raises or drops the request.
"""
from typing import Any

_VALID_ACTIONS = {"retrieve", "direct_answer", "tool_use"}
_VALID_TOOL_CHOICES = {"web_search", "sandbox", None}


def validate_plan(plan: Any) -> dict:
    if not isinstance(plan, dict):
        return {}
    validated = dict(plan)
    if validated.get("action") not in _VALID_ACTIONS:
        validated["action"] = "retrieve"
    if validated.get("tool_choice") not in _VALID_TOOL_CHOICES:
        validated["tool_choice"] = None
    return validated

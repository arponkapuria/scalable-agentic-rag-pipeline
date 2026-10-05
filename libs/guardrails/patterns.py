"""
Regex pattern lists used by the input and output guardrail checks. Each entry pairs a compiled pattern with a short label so a block can be logged with a reason. Patterns match multi-word phrases specific to an override or leak attempt, never a single common word, so legitimate questions aren't blocked by accident.
"""
import re

_F = re.IGNORECASE

# Input: blocks prompt injection, jailbreak, and persona-override attempts
INPUT_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|above|prior)\s+instructions", _F), "override_instructions"),
    (re.compile(r"(disregard|forget)\s+(your\s+|all\s+)?(previous\s+)?(instructions|rules|guidelines)", _F), "override_instructions"),
    (re.compile(r"reveal\s+(your\s+|the\s+)?(system\s+prompt|hidden\s+instructions)", _F), "prompt_extraction"),
    (re.compile(r"(what\s+(is|are)|show\s+me)\s+your\s+(system\s+prompt|initial\s+instructions)", _F), "prompt_extraction"),
    (re.compile(r"act\s+as\s+(if\s+you\s+(are|were)\s+)?an?\s+\w+\s+(with\s+no|without)\s+restrictions", _F), "persona_override"),
    (re.compile(r"you\s+are\s+now\s+(in\s+)?(dan|developer)\s+mode", _F), "persona_override"),
    (re.compile(r"\bdo\s+anything\s+now\b", _F), "persona_override"),
)

# Output: blocks the model repeating its own system prompt or internal instructions
OUTPUT_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"you\s+are\s+a\s+helpful\s+(enterprise\s+)?assistant", _F), "system_prompt_leak"),
    (re.compile(r"my\s+(system\s+)?instructions\s+(are|include|say)", _F), "system_prompt_leak"),
    (re.compile(r"i\s+(was|am)\s+instructed\s+to", _F), "system_prompt_leak"),
    (re.compile(r"\[?system\s*prompt\]?\s*:", _F), "system_prompt_leak"),
)

# A request this long is either abuse or a mistake, so it's declined before reaching the planner. Generous on purpose: real questions, even long ones, run a few hundred characters.
MAX_INPUT_CHARS = 4000
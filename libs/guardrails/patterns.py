"""
Guardrail pattern lists. Each entry is (compiled pattern, short label) so a
block can be logged with WHY it fired, not just THAT it fired.

Scoping rule followed throughout: patterns match multi-word phrases specific
to an override/leak attempt, never a single common word — a paper can
legitimately discuss "ignoring padding tokens" or a person named "Dan"
without tripping anything here. tests/test_guardrails.py's must-not-block
list is the actual spec for how narrow these need to be; a pattern that
fails a case there gets narrowed, not the test loosened.
"""
import re

_F = re.IGNORECASE

# --- Input: prompt injection / jailbreak / persona-override attempts ---
INPUT_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|above|prior)\s+instructions", _F), "override_instructions"),
    (re.compile(r"(disregard|forget)\s+(your\s+|all\s+)?(previous\s+)?(instructions|rules|guidelines)", _F), "override_instructions"),
    (re.compile(r"reveal\s+(your\s+|the\s+)?(system\s+prompt|hidden\s+instructions)", _F), "prompt_extraction"),
    (re.compile(r"(what\s+(is|are)|show\s+me)\s+your\s+(system\s+prompt|initial\s+instructions)", _F), "prompt_extraction"),
    (re.compile(r"act\s+as\s+(if\s+you\s+(are|were)\s+)?an?\s+\w+\s+(with\s+no|without)\s+restrictions", _F), "persona_override"),
    (re.compile(r"you\s+are\s+now\s+(in\s+)?(dan|developer)\s+mode", _F), "persona_override"),
    (re.compile(r"\bdo\s+anything\s+now\b", _F), "persona_override"),
)

# --- Output: the model repeating its own system prompt or internal state ---
OUTPUT_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"you\s+are\s+a\s+helpful\s+(enterprise\s+)?assistant", _F), "system_prompt_leak"),
    (re.compile(r"my\s+(system\s+)?instructions\s+(are|include|say)", _F), "system_prompt_leak"),
    (re.compile(r"i\s+(was|am)\s+instructed\s+to", _F), "system_prompt_leak"),
    (re.compile(r"\[?system\s*prompt\]?\s*:", _F), "system_prompt_leak"),
)

# A single request pasting in tens of thousands of characters is either
# abuse (token-budget exhaustion) or a mistake — either way, decline before
# it reaches the planner. Generous on purpose: real questions, even long
# multi-part ones, run a few hundred characters.
MAX_INPUT_CHARS = 4000

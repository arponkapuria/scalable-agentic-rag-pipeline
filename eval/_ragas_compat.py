"""
Compatibility shim for a real, verified bug in ragas 0.4.3's own package
init: `ragas/__init__.py` -> `ragas/llms/base.py` unconditionally imports
`ChatVertexAI` from `langchain_community.chat_models.vertexai`, a
submodule that no longer exists in current `langchain-community`
releases (that integration moved to the standalone
`langchain-google-vertexai` package; `langchain-community` is
mid-deprecation and dropped it). This breaks a plain `import ragas` on a
clean `pip install ragas`/`uv sync --group eval` as of 0.4.3 — verified
directly by installing it and reading the traceback, not assumed.

This project never uses VertexAI anywhere, so the fix is to register a
dummy module at that import path before ragas imports it. This is a
workaround for ragas's OWN broken transitive import, not a statement
about anything in this codebase — delete this shim (and every
`import eval._ragas_compat  # noqa: F401` line that references it) once
ragas ships a fix upstream (their own pin on langchain-community needs
tightening, or that import needs to move behind a try/except the way
most optional-integration imports do — most of ragas/llms/base.py's
other provider integrations already do this).

MUST be imported before anything else imports `ragas`. Every eval module
that touches ragas does `import eval._ragas_compat  # noqa: F401` as its
first import, specifically before any `from ragas... import ...` line.
"""
import sys
import types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # pragma: no cover — never actually instantiated
        """Dummy stand-in. ragas's own __init__ chain imports this name
        at module-load time; nothing in this project ever constructs
        it (no VertexAI usage anywhere in OmniRAG)."""

    _stub.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _stub
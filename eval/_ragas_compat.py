"""Shim for ragas 0.4.3, whose import chain needs a module that newer langchain-community removed.

Import this before anything that imports ragas, then delete it once ragas fixes the import upstream.
"""
import sys
import types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # pragma: no cover
        """Empty placeholder so ragas can import the name; it is never used."""

    _stub.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _stub
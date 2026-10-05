"""
Regex-based guardrails for the RAG pipeline. Input filtering blocks unsafe requests before any LLM call, output filtering blocks the model from leaking its own system prompt, and plan validation checks the planner's JSON output before it is used downstream. No model calls are involved, so every decision is fast and inspectable as a plain pattern list.
"""
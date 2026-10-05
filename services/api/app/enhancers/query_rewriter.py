"""
Multi-turn query rewriter: resolves coreferences ("it", "that", "he") using conversation history, producing a standalone search query.
"""
from typing import List, Dict
from services.api.app.clients.llm.factory import llm_client
from libs.utils.model_router import route_model

SYSTEM_PROMPT = """
You are a Query Rewriter. 
Your task is to rewrite the latest user question to be a standalone search query, 
resolving coreferences (he, she, it, they) using the conversation history.

History:
{history}

Latest Question: {question}

Output ONLY the rewritten question, as a short search query (under 20
words). Do not answer the question. Do not add explanation, formatting,
or markdown. If no rewriting is needed, output the latest question as is.
"""

_MAX_REWRITTEN_QUERY_CHARS = 200  # backstop if a model writes a full answer instead of a short query


async def rewrite_query(question: str, history: List[Dict[str, str]]) -> str:
    """Rewrites the latest question into a standalone search query using conversation history.

    Args:
        question: The user's latest message.
        history: Prior conversation turns, as {"role", "content"} dicts.

    Returns:
        The rewritten query, or the original question on no history, an implausibly long
        rewrite, or an LLM failure.
    """
    if not history:
        return question

    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
    prompt = SYSTEM_PROMPT.format(history=history_str, question=question)

    try:
        rewritten = await llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=64,
            model=route_model(question),
        )
        rewritten = rewritten.strip()
        return question if len(rewritten) > _MAX_REWRITTEN_QUERY_CHARS else rewritten
    except Exception:
        return question
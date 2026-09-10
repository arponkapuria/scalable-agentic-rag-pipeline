from typing import List, Dict
from services.api.app.clients.llm.factory import llm_client

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

# Hard backstop in case a model ignores the prompt anyway (observed:
# gpt-oss-120b returned a full multi-paragraph answer instead of a
# rewritten query for one 'What is X?' input) — a rewritten "query" this
# long is almost certainly a free-written answer, not a search query, and
# using it for retrieval would badly pollute the embedding/BM25 search.
# Falling back to the original question is safer than either passing
# through garbage or failing the request.
_MAX_REWRITTEN_QUERY_CHARS = 200

async def rewrite_query(question: str, history: List[Dict[str, str]]) -> str:
    """
    Uses LLM to de-contextualize the query for better Vector Search.
    """
    if not history:
        return question

    # Format history into a string
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
    
    prompt = SYSTEM_PROMPT.format(history=history_str, question=question)

    try:
        rewritten = await llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # Strict logic
            max_tokens=64,    # A search query is a handful of words; caps
                               # runaway generation even if the model
                               # ignores "output only the query."
        )
        rewritten = rewritten.strip()

        if len(rewritten) > _MAX_REWRITTEN_QUERY_CHARS:
            return question

        return rewritten
    except Exception as e:
        # Fallback to original query if LLM fails
        return question
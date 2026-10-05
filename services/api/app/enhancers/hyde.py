"""
HyDE (Hypothetical Document Embeddings): generates a fake but plausibly-worded answer to embed instead of the raw question, for a closer vector match. Opt-in per request, off by default.
"""
from services.api.app.clients.llm.factory import llm_client

SYSTEM_PROMPT = """
You are a helpful assistant. 
Write a hypothetical paragraph that answers the user's question. 
It does not need to be factually correct, but it must use the correct vocabulary and structure 
that a relevant document would have.

Question: {question}
"""


async def generate_hypothetical_document(question: str) -> str:
    """Generates a hypothetical answer document to improve vector similarity search.

    Args:
        question: The user's question.

    Returns:
        A generated paragraph, or the original question unchanged if generation fails.
    """
    try:
        return await llm_client.chat_completion(
            messages=[{"role": "system", "content": SYSTEM_PROMPT.format(question=question)}],
            temperature=0.7,
        )
    except Exception:
        return question
from services.api.app.clients.llm.factory import llm_client

SYSTEM_PROMPT = """
You are a helpful assistant. 
Write a hypothetical paragraph that answers the user's question. 
It does not need to be factually correct, but it must use the correct vocabulary and structure 
that a relevant document would have.

Question: {question}
"""

async def generate_hypothetical_document(question: str) -> str:
    """
    Generates a fake document to improve vector similarity search.
    """
    try:
        hypothetical_doc = await llm_client.chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(question=question)},
            ],
            temperature=0.7 # Higher temp to generate diverse vocabulary
        )
        return hypothetical_doc
    except Exception:
        return question # Fallback
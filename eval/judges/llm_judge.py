from pydantic import BaseModel
import json
from services.api.app.clients.llm.factory import llm_client

class Grade(BaseModel):
    score: int
    reasoning: str

JUDGE_PROMPT = """
You are an impartial judge evaluating a RAG system.
You will be given a Question, a Ground Truth Answer, and the System's Answer.

Rate the System's Answer on a scale of 1 to 5:
1: Completely wrong or hallucinated.
3: Partially correct but missing key details.
5: Perfect, comprehensive, and matches Ground Truth logic.

Output JSON only: {{"score": int, "reasoning": "string"}}

Question: {question}
Ground Truth: {ground_truth}
System Answer: {system_answer}
"""

async def grade_answer(question: str, ground_truth: str, system_answer: str) -> Grade:
    """
    Calls the LLM to grade a single QA pair.
    """
    
    try:
        response_text = await llm_client.chat_completion(
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                question=question,
                ground_truth=ground_truth,
                system_answer=system_answer
            )}],
            temperature=0.0
        )
        
        # Parse JSON output
        data = json.loads(response_text)
        return Grade(**data)
        
    except Exception as e:
        return Grade(score=0, reasoning=f"Judge Error: {e}")
import json
import logging
from services.api.app.agents.state import AgentState
from services.api.app.clients.llm.factory import llm_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a RAG Planning Agent.
Analyze the User Query and Conversation History.

Decide the next step:
1. If the user greets (Hello/Hi), output "direct_answer".
2. If the user asks a specific question requiring data, output "retrieve".
3. If the user asks for code execution, output "tool_use".

Output JSON format ONLY:
{
    "action": "retrieve" | "direct_answer" | "tool_use",
    "refined_query": "The standalone search query",
    "reasoning": "Why you chose this action",
    "tool_choice": "web_search" | null,
    "tool_input": "The exact input to pass to the tool" | null
}
"""

async def planner_node(state: AgentState) -> dict:
    """
    Decides the path through the LangGraph.
    """
    logger.info("Planner Node: Analyzing query...")
    
    # Extract latest user message
    # state['messages'] is a list of dicts or objects
    last_message = state["messages"][-1]

    if isinstance(last_message, dict):
        user_query = last_message.get("content", "")
    else:
        user_query = getattr(last_message, "content", "")

    # Call LLM to plan
    try:
        response_text = await llm_client.chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query}
            ],
            temperature=0.0, # Deterministic planning
            json_mode=True
        )
        
        # Parse JSON
        plan = json.loads(response_text)
        
        logger.info(f"Plan derived: {plan['action']}")
        
        # Update State
        return {
            "current_query": plan.get("refined_query", user_query),
            "plan": [plan["reasoning"]],
            "action": plan.get("action", "retrieve"),
            "tool_choice": plan.get("tool_choice", ""),   
            "tool_input": plan.get("tool_input", "")  
        }
        
    except Exception as e:
        # Pre-existing, non-fatal: json_mode occasionally returns content
        # that fails to parse (root cause not fully pinned down — see
        # PROGRESS.md). Fallback below always keeps the request working,
        # so this stays a warning, not an error.
        logger.warning(f"Planning failed, defaulting to retrieve: {e}")
        # Fallback: Assume we need to search
        return {
            "current_query": user_query,
            "plan": ["Error in planning, defaulting to retrieval."],
            "action": "retrieve"
        }
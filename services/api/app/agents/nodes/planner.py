"""
Decides the next action: retrieve from the corpus, call a tool, or answer directly. Also flags yes/no existence-check questions, which change how a zero-hit result is worded downstream.
"""
import json
import logging
from services.api.app.agents.state import AgentState
from libs.guardrails.plan_schema import validate_plan
from services.api.app.clients.llm.factory import llm_client
from libs.utils.model_router import route_model

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a RAG Planning Agent.
Analyze the User Query and Conversation History.

Decide the next step:
1. If the user greets (Hello/Hi), says thanks, or makes pure small talk
   with NO factual/informational content to look up, output "direct_answer".
2. If the user asks ANY question seeking a fact, data, current information,
   or real-world state (this includes things the document corpus is
   unlikely to cover, e.g. weather, news, prices, current events), output
   "retrieve". Do NOT reason about whether the corpus probably has the
   answer — that is retrieval's job to determine, not yours. Retrieval
   finding nothing is what correctly triggers a web-search offer to the
   user; a plain "direct_answer" for this skips that flow entirely and
   lets the model improvise a suggestion on its own instead (e.g.
   "try a weather app"), which is NOT the intended behavior.

   For "retrieve" specifically, also set "is_existence_check": true if
   the question is a yes/no question asking whether the document(s)
   mention, describe, use, or contain something (e.g. "did the paper use
   ResNet", "is X mentioned", "does the author discuss Y") — as opposed
   to an open-ended informational question ("what does the paper say
   about X", "how does X work"). Default to false if uncertain.
3. If the user asks for code execution, output "tool_use" with tool_choice="sandbox".
4. Web search is NEVER triggered on your own initiative. It is only used
   when: the MOST RECENT assistant message in the conversation history
   explicitly asked the user whether to search the web for something,
   AND the user's current message is a clear affirmative response to
   that specific offer (e.g. "yes", "please do", "go ahead"). In that
   case ONLY, output "tool_use" with tool_choice="web_search" and
   tool_input set to the topic being searched for, read from the
   assistant's offer message. Otherwise, do not use web_search.

Output JSON format ONLY:
{
    "action": "retrieve" | "direct_answer" | "tool_use",
    "refined_query": "The standalone search query",
    "reasoning": "Why you chose this action",
    "tool_choice": "web_search" | "sandbox" | null,
    "tool_input": "The exact input to pass to the tool" | null,
    "is_existence_check": true | false
}
"""


async def planner_node(state: AgentState) -> dict:
    """Runs the planner LLM call and updates state with the chosen action.

    Args:
        state: Current agent state.

    Returns:
        Partial update with "current_query", "plan", "action", "tool_choice", "tool_input",
        and "is_existence_check".
    """
    logger.info("Planner Node: Analyzing query...")

    # current_query is the coreference-resolved, standalone query; messages[-1] is the raw
    # user message (kept for real conversation history), used only as a fallback.
    user_query = state.get("current_query") or ""
    if not user_query:
        last_message = state["messages"][-1]
        user_query = last_message.get("content", "") if isinstance(last_message, dict) else getattr(last_message, "content", "")

    # Last 2 turns — enough to detect a bare "yes" confirming a web-search offer.
    recent_history = (state.get("messages") or [])[-2:]
    history_messages = [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        for m in recent_history if isinstance(m, dict)
    ]

    try:
        response_text = await llm_client.chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *history_messages,
                {"role": "user", "content": user_query}
            ],
            temperature=0.0,
            json_mode=True,
            model=route_model(user_query),
        )

        plan = validate_plan(json.loads(response_text))
        logger.info(f"Plan derived: {plan.get('action')}")

        # `or` falls back on both a missing key and an explicit JSON null.
        action = plan.get("action") or "retrieve"
        # Only tool turns adopt the planner's rewrite; retrieve/direct_answer keep the
        # coreference-resolved query, which retrieval scores better against.
        return {
            "current_query": (plan.get("refined_query") or user_query) if action == "tool_use" else user_query,
            "plan": [plan["reasoning"]],
            "action": action,
            "tool_choice": plan.get("tool_choice") or "",
            "tool_input": plan.get("tool_input") or "",
            "is_existence_check": bool(plan.get("is_existence_check")),
        }

    except Exception as e:
        logger.warning(f"Planning failed, defaulting to retrieve: {e}")
        return {
            "current_query": user_query,
            "plan": ["Error in planning, defaulting to retrieval."],
            "action": "retrieve",
            "is_existence_check": False,
        }
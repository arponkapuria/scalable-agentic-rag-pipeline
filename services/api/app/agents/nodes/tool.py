"""
Executes the planner's selected tool — vector search, web search, or the code sandbox — and returns its output for the responder.
"""
import logging
from services.api.app.agents.state import AgentState
from services.api.app.tools.web_search import web_search_tool
from services.api.app.tools.sandbox import run_python_code
from services.api.app.tools.vector_search import search_vector_tool

logger = logging.getLogger(__name__)


async def tool_node(state: AgentState) -> dict:
    """Runs the tool named in state["tool_choice"] and records its output.

    Args:
        state: Current agent state. Reads "tool_choice", "tool_input", "corpus_id",
            and "current_query" (fallback search term for web_search).

    Returns:
        Partial update with "messages", "tool_used", and "tool_result".
    """
    plan_data = state.get("plan", [])
    if not plan_data:
        return {"messages": [{"role": "system", "content": "No tool selected."}]}

    tool_name = state.get("tool_choice") or "web_search"
    tool_input = state.get("tool_input") or ""

    result = ""

    if tool_name == "vector_search":
        logger.info(f"Executing Vector Search: {tool_input}")
        result = await search_vector_tool(tool_input, state["corpus_id"])

    elif tool_name == "web_search":
        search_query = tool_input or state.get("current_query") or ""
        logger.info(f"Executing Web Search: {search_query}")
        result = await web_search_tool(search_query)

    elif tool_name == "sandbox":
        logger.info(f"Executing Python Sandbox: {tool_input}")
        result = await run_python_code(tool_input)

    else:
        result = "Unknown tool requested."

    return {
        "messages": [{"role": "user", "content": f"Tool Output: {result}"}],
        "tool_used": tool_name,
        "tool_result": result,
    }
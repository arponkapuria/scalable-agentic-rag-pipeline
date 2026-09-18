import logging
from services.api.app.agents import state
from services.api.app.agents.state import AgentState
from services.api.app.tools.web_search import web_search_tool
from services.api.app.tools.sandbox import run_python_code
from services.api.app.tools.vector_search import search_vector_tool

logger = logging.getLogger(__name__)

async def tool_node(state: AgentState) -> dict:
    """
    Executes the tool specified in the plan.
    """
    # Get the last message or plan to see what tool to call
    # In a real implementation, the Planner outputs a structured tool call.
    # Here we simplify based on the 'plan' field.
    
    plan_data = state.get("plan", [])
    if not plan_data:
        return {"messages": [{"role": "system", "content": "No tool selected."}]}

    # Assume Planner passed specific instruction in state (simplified)
    # Real implementations use OpenAI function calling API or JSON parsing
    tool_name = state.get("tool_choice") or "web_search"
    tool_input = state.get("tool_input") or ""
    
    result = ""

    if tool_name == "vector_search":
        logger.info(f"Executing Vector Search: {tool_input}")
        result = await search_vector_tool(tool_input, state["corpus_id"])
    
    elif tool_name == "web_search":
        # Fallback if the planner's tool_input extraction came back empty
        # (e.g. the offer's echoed topic wasn't cleanly parsed) — better
        # than searching for an empty string.
        search_query = tool_input or state.get("current_query") or ""
        logger.info(f"Executing Web Search: {search_query}")
        result = await web_search_tool(search_query)

    elif tool_name == "sandbox":
        logger.info(f"Executing Python Sandbox: {tool_input}")
        result = await run_python_code(tool_input)
        
    else:
        result = "Unknown tool requested."

    # Return the observation. tool_used marks which tool actually ran —
    # Phase 5's L2 semantic cache is gated on this: only vector_search is
    # cache-eligible there, web_search (staleness) and sandbox (numeric
    # precision) are never L2-cached, matching the locked design.
    # tool_result carries the raw output separately (Phase 6 follow-up) —
    # see generate_node's sandbox branch for why.
    return {
        "messages": [
            {"role": "user", "content": f"Tool Output: {result}"}
        ],
        "tool_used": tool_name,
        "tool_result": result,
    }
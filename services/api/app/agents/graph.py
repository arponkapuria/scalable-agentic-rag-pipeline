"""
Builds the LangGraph state machine for one chat turn.
"""
from langgraph.graph import StateGraph, END
from services.api.app.agents.state import AgentState
from services.api.app.agents.nodes.retriever import retrieve_node
from services.api.app.agents.nodes.responder import generate_node
from services.api.app.agents.nodes.planner import planner_node
from services.api.app.agents.nodes.tool import tool_node

workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("responder", generate_node)
workflow.add_node("tool", tool_node)


def route_from_planner(state: AgentState) -> str:
    """Picks the next node from the planner's chosen action.

    Args:
        state: Current agent state.

    Returns:
        "responder", "tool", or "retriever".
    """
    action = state.get("action") or "retrieve"
    if action == "direct_answer":
        return "responder"
    elif action == "tool_use":
        return "tool"
    return "retriever"


workflow.set_entry_point("planner")

workflow.add_conditional_edges(
    "planner",
    route_from_planner,
    {"retriever": "retriever", "responder": "responder", "tool": "tool"}
)

workflow.add_edge("tool", "responder")
workflow.add_edge("retriever", "responder")
workflow.add_edge("responder", END)

agent_app = workflow.compile()
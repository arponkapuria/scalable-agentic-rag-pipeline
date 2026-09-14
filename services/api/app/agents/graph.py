from langgraph.graph import StateGraph, END
from services.api.app.agents.state import AgentState
from services.api.app.agents.nodes.retriever import retrieve_node
from services.api.app.agents.nodes.responder import generate_node
from services.api.app.agents.nodes.planner import planner_node
from services.api.app.agents.nodes.tool import tool_node

# Initialize the Graph
workflow = StateGraph(AgentState)

# 1. Define Nodes (The Logic Steps)
# These functions (imported above) will be implemented in the 'nodes/' folder next
workflow.add_node("planner", planner_node)       # Rewrites query / Decides steps
workflow.add_node("retriever", retrieve_node)    # Hits Qdrant (hybrid dense+sparse+RRF)
workflow.add_node("responder", generate_node)    # Calls Ray Serve LLM
workflow.add_node("tool", tool_node)

# 2. Routing function
def route_from_planner(state: AgentState) -> str:
    action = state.get("action") or "retrieve"
    if action == "direct_answer":
        return "responder"
    elif action == "tool_use":
        return "tool"
    return "retriever"

# 3. Edges
# Conditional routing based on planner decision:
#   retrieve      → retriever → responder → END
#   direct_answer → responder → END
#   tool_use      → tool → responder → END
workflow.set_entry_point("planner")

workflow.add_conditional_edges(
    "planner",
    route_from_planner,
    {
        "retriever": "retriever",
        "responder": "responder",
        "tool": "tool"
    }
)

workflow.add_edge("tool", "responder")
workflow.add_edge("retriever", "responder")
workflow.add_edge("responder", END)  # In a more complex agent, we could loop back if answer is bad

# 4. Compile the Graph
# This creates the runnable application
agent_app = workflow.compile()
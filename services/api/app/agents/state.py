"""
Shared state passed between every node in the LangGraph agent.
"""
from typing import TypedDict, Annotated, List
import operator


class AgentState(TypedDict):
    """State object passed between nodes in the LangGraph."""
    messages: Annotated[List[dict], operator.add]
    documents: List[str]
    current_query: str
    plan: List[str]
    action: str  # "retrieve" | "direct_answer" | "tool_use"
    tool_choice: str  # "web_search" | "sandbox" | ""
    tool_input: str
    corpus_id: str  # multi-tenant scope, set from the session cookie only
    tool_used: str
    backend_used: str
    sources: list[str]
    model_used: str  # actual model id that answered, empty if no LLM call was made
    tool_result: str  # raw tool output, used for the sandbox's verbatim echo
    is_existence_check: bool  # yes/no question about document content vs. an open-ended question
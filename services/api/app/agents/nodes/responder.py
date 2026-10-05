"""
Synthesizes the final answer. Branches by action: retrieval-grounded answers get a RAG prompt with a refusal instruction, conversational answers get a plain prompt, sandbox output is echoed verbatim with no LLM call.
"""
import logging

from services.api.app.agents.state import AgentState
from services.api.app.clients.llm.factory import llm_client
from services.api.app.config import settings
from libs.utils.model_router import route_model
from libs.utils.refusal import TRUNCATION_NOTE, is_refusal
from libs.guardrails.filters import guard_output

logger = logging.getLogger(__name__)


async def generate_node(state: AgentState) -> dict:
    """Synthesizes the final answer, branching by action.

    Args:
        state: Current agent state. Reads "current_query", "documents", "action", and
            "tool_choice"/"tool_result" for the sandbox path.

    Returns:
        Partial update with "messages", "backend_used", "model_used", after the output
        guardrail has run.
    """
    query = state["current_query"]
    documents = state["documents"] or []
    action = state.get("action") or "retrieve"

    if action == "retrieve":
        result = await _answer_from_documents(query, documents, bool(state.get("is_existence_check")))
    elif action == "tool_use" and state.get("tool_choice") == "sandbox":
        # Deterministic echo, no LLM call — avoids a model mis-transcribing stdout or
        # accidentally emitting its own tool-call format when asked to narrate tool output.
        result = {
            "messages": [{"role": "assistant", "content": _echo_sandbox_result(state.get("tool_result") or "")}],
            "backend_used": "none",
            "model_used": "",
        }
    else:
        result = await _answer_conversationally(state, query)

    return guard_output(result)


def _echo_sandbox_result(tool_result: str) -> str:
    """Formats the sandbox's raw output for display.

    Args:
        tool_result: Prefixed output from tools/sandbox.py.

    Returns:
        Successful output in a code block; error strings returned as-is.
    """
    if not tool_result:
        return "The code sandbox didn't return a result."
    if tool_result.startswith("Output:"):
        stdout = tool_result[len("Output:"):].strip()
        return f"Here's the result:\n\n```\n{stdout}\n```"
    return tool_result


def _offer_web_search(query: str) -> str:
    """Builds the standard web-search offer text, shared by every refusal path so the planner only needs to recognize one offer format.

    Args:
        query: The question that couldn't be answered from the corpus.

    Returns:
        The offer text.
    """
    return (
        f"I don't have information about \"{query}\" in my documents. "
        "Would you like me to search the web for it?"
    )


def _existence_check_negative() -> str:
    """Builds the zero-hit answer for existence-check questions.

    Returns:
        A fixed negative answer, not a web-search offer — this is a genuine, cacheable answer.
    """
    return "No — that isn't mentioned in the documents."


def _flag_truncation(answer: str) -> str:
    """Appends a visible note if the last LLM call hit the token cap.

    Args:
        answer: The generated answer text.

    Returns:
        The answer, with TRUNCATION_NOTE appended if truncated.
    """
    if getattr(llm_client, "last_finish_reason", "") == "length":
        return answer + TRUNCATION_NOTE
    return answer


async def _answer_from_documents(query: str, documents: list[str], is_existence_check: bool) -> dict:
    """Generates a RAG answer grounded in retrieved documents, or a deterministic non-answer if there's nothing to ground it in.

    Args:
        query: The user's question.
        documents: Retrieved context chunks.
        is_existence_check: Changes the wording of a zero-hit or refused answer.

    Returns:
        A dict with "messages", "backend_used", "model_used".
    """
    if not documents:
        if is_existence_check:
            logger.info("No documents retrieved for an existence-check query.")
            return {
                "messages": [{"role": "assistant", "content": _existence_check_negative()}],
                "backend_used": "none",
                "model_used": "",
            }
        logger.info("No documents retrieved — offering web search instead.")
        return {
            "messages": [{"role": "assistant", "content": _offer_web_search(query)}],
            "backend_used": "none",
            "model_used": "",
        }

    context_str = "\n\n".join(documents)
    routed_model = route_model(query, len(context_str)) if settings.ENABLE_MODEL_ROUTING else None
    if routed_model:
        logger.info(f"Responder routed to model: {routed_model}")

    answer = await llm_client.chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful Enterprise Assistant. "
                    "Always cite sources using [Source: Filename]. "
                    "If the answer is not in the context, say "
                    "'I don't have that information in my documents.' "
                    "Be concise and professional."
                )
            },
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion:\n{query}"}
        ],
        temperature=0.3,
        model=routed_model,
        max_tokens=settings.ANSWER_MAX_TOKENS,
    )
    model_used = getattr(llm_client, "last_model_used", "") or routed_model or ""

    # Retrieval found hits, but they were irrelevant enough that the LLM refused anyway —
    # normalize to the same offer/negative as the zero-hit case above.
    if is_refusal(answer):
        content = _existence_check_negative() if is_existence_check else _offer_web_search(query)
        return {
            "messages": [{"role": "assistant", "content": content}],
            "backend_used": "none",
            "model_used": "",
        }

    backend_used = getattr(llm_client, "last_backend_used", "") or llm_client.__class__.__name__
    return {
        "messages": [{"role": "assistant", "content": _flag_truncation(answer)}],
        "backend_used": backend_used,
        "model_used": model_used,
    }


async def _answer_conversationally(state: AgentState, query: str) -> dict:
    """Generates a plain conversational answer — for greetings/chitchat, and for narrating tool output (web_search).

    Args:
        state: Current agent state, used for recent conversation history.
        query: The user's current message.

    Returns:
        A dict with "messages", "backend_used", "model_used".
    """
    recent_history = (state.get("messages") or [])[-4:]
    routed_model = route_model(query) if settings.ENABLE_MODEL_ROUTING else None
    if routed_model:
        logger.info(f"Responder (conversational) routed to model: {routed_model}")

    answer = await llm_client.chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful, friendly assistant. Respond naturally "
                    "and concisely. If recent messages include tool output, use "
                    "it to answer the user's question. If a tool's output "
                    "indicates a failure, timeout, or connection error, report "
                    "that honestly instead of claiming you lack the capability."
                )
            },
            *[{"role": m.get("role", "user"), "content": m.get("content", "")} for m in recent_history],
            {"role": "user", "content": query},
        ],
        temperature=0.5,
        model=routed_model,
        max_tokens=settings.ANSWER_MAX_TOKENS,
    )

    backend_used = getattr(llm_client, "last_backend_used", "") or llm_client.__class__.__name__
    model_used = getattr(llm_client, "last_model_used", "") or routed_model or ""
    return {
        "messages": [{"role": "assistant", "content": _flag_truncation(answer)}],
        "backend_used": backend_used,
        "model_used": model_used,
    }
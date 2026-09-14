import logging

from services.api.app.agents.state import AgentState
from services.api.app.clients.llm.factory import llm_client
from services.api.app.config import settings
from libs.utils.model_router import route_model

logger = logging.getLogger(__name__)


async def generate_node(state: AgentState) -> dict:
    """
    Synthesizes the final answer using retrieved documents.
    """
    query = state["current_query"]
    documents = state["documents"] or []

    # Deterministic short-circuit for "retrieval genuinely found nothing" —
    # NOT for direct_answer/tool_use paths, which legitimately have empty
    # `documents` (see agents/graph.py's routing). Added after a real,
    # observed failure: the LLM was called with zero context anyway and
    # fabricated both an answer and a citation to a filename that doesn't
    # exist in the corpus, ignoring its own "say you don't have it"
    # system-prompt instruction outright. Don't rely on prompt-following
    # for something this consequential when a deterministic check costs
    # nothing and also saves an LLM call.
    if state.get("action") == "retrieve" and not documents:
        logger.info("No documents retrieved — skipping LLM call, returning fixed no-context answer.")
        return {
            "messages": [{"role": "assistant", "content": "I don't have that information in my documents."}],
            "backend_used": "none",
        }

    # Construct Context String
    context_str = "\n\n".join(documents)

    # Heuristic model-tier routing (see model_router.py) — only applies
    # here, not to planner/query-rewriter/graph-extraction. `routed_model`
    # is None when routing is disabled, which chat_completion treats
    # identically to "no pin" (falls back to the backend's own priority
    # list/round-robin).
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
            {
                "role": "user",
                "content": f"Context:\n{context_str}\n\nQuestion:\n{query}"
            }
        ],
        temperature=0.3,
        model=routed_model,
    )

    # backend_used: FailoverLLMClient (LLM_BACKEND=api) tracks which of
    # Groq/OpenRouter actually answered; manual-select backends (ollama/
    # vllm_local/vllm_modal) have no failover concept, so their class name
    # is the whole story. Phase 5's cache stores this alongside the answer.
    backend_used = getattr(llm_client, "last_backend_used", "") or llm_client.__class__.__name__

    # Return dictionary to update state (add the AI message)
    return {
        "messages": [{"role": "assistant", "content": answer}],
        "backend_used": backend_used,
    }
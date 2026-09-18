import logging

from services.api.app.agents.state import AgentState
from services.api.app.clients.llm.factory import llm_client
from services.api.app.config import settings
from libs.utils.model_router import route_model
from libs.utils.refusal import is_refusal

logger = logging.getLogger(__name__)


async def generate_node(state: AgentState) -> dict:
    """
    Synthesizes the final answer. Branches by action — direct_answer and
    tool_use paths never have `documents` populated by design (retrieval
    never runs for them), so they must NOT share the RAG system prompt,
    which instructs the model to refuse whenever context is empty. Bug
    fixed here: before this branch existed, a plain "hey" (routed
    direct_answer, correctly) still got "I don't have that information in
    my documents." — the model was just following the RAG prompt's own
    instruction literally against an always-empty context, for a turn
    that was never supposed to have any.
    """
    query = state["current_query"]
    documents = state["documents"] or []
    action = state.get("action") or "retrieve"

    if action == "retrieve":
        return await _answer_from_documents(query, documents, bool(state.get("is_existence_check")))
    elif action == "tool_use" and state.get("tool_choice") == "sandbox":
        # Deterministic echo, no LLM call (Phase 6 follow-up) — replaces
        # asking the LLM to narrate the sandbox's stdout. Two real
        # problems this avoids, both live-observed: (1) tool-tuned models
        # (the gpt-oss family) can slip into emitting their OWN native
        # tool-call format when a system prompt talks about "tool
        # output," which Groq's API then rejects outright ('Tool choice
        # is none, but model called a tool') — burning through the whole
        # model fallback chain on retries before anything succeeds, pure
        # wasted latency/cost even when it eventually works; (2) there's
        # no real reason to risk the LLM mis-transcribing/paraphrasing a
        # program's actual output when the real, verbatim result is
        # already sitting in state — narrating it is pure downside here,
        # unlike web_search, where synthesizing results is genuinely
        # useful. See generate_node's docstring below for tool_result.
        return {
            "messages": [{"role": "assistant", "content": _echo_sandbox_result(state.get("tool_result") or "")}],
            "backend_used": "none",
            "model_used": "",
        }
    else:
        # direct_answer (greetings/chitchat) and tool_use for any OTHER
        # tool (e.g. web_search) go through a plain conversational prompt
        # — no "cite sources"/"refuse if no context" instructions, since
        # neither concept applies here.
        return await _answer_conversationally(state, query)


def _echo_sandbox_result(tool_result: str) -> str:
    """Sandbox output already comes prefixed (see tools/sandbox.py):
    "Output:\n...", "Execution Error:\n...", "Sandbox Error: ...", or
    "Sandbox Connection Failed: ...". Wrapped in a code block for
    successful output specifically — an error/failure string isn't code,
    doesn't need the formatting, and reads better as plain text."""
    if not tool_result:
        return "The code sandbox didn't return a result."
    if tool_result.startswith("Output:"):
        stdout = tool_result[len("Output:"):].strip()
        return f"Here's the result:\n\n```\n{stdout}\n```"
    return tool_result  # error/failure strings — already human-readable as-is


def _offer_web_search(query: str) -> str:
    """Shared by both refusal paths below — the deterministic zero-hit
    short-circuit AND the post-generation check for when retrieval found
    SOME (irrelevant) hits but the LLM still correctly concluded it
    couldn't answer from them. Both funnel into the same offer text so
    the behavior is consistent regardless of which path triggered it, and
    so planner.py's confirmation-detection rule only needs to recognize
    one phrasing."""
    return (
        f"I don't have information about \"{query}\" in my documents. "
        "Would you like me to search the web for it?"
    )


def _existence_check_negative() -> str:
    """Zero-hit answer for is_existence_check questions (Phase 6
    follow-up) — "did X use Y" / "is Z mentioned" style questions, where
    the absence of matching chunks IS itself a complete, correct answer.
    Deliberately does NOT offer a web search — the question is
    specifically about this document's content, so an external search
    wouldn't answer what was actually asked. Not run through
    is_refusal()/REFUSAL_PREFIXES: this is a genuine answer, not a
    non-answer, so it's fine to cache like any other response.

    Known limitation, not solved here: planner.py's is_existence_check
    classification can't distinguish "is X mentioned in THIS document"
    from a yes/no question that's really general-knowledge (e.g. "is
    Python interpreted?"), where a flat "No, not mentioned" would be a
    confidently WRONG answer instead of just an unhelpful one. In
    practice, for this project's literature-survey use case, most yes/no
    questions genuinely are about document content — but this is a real
    trade-off, not a fully solved case."""
    return "No — that isn't mentioned in the documents."


async def _answer_from_documents(query: str, documents: list[str], is_existence_check: bool) -> dict:
    # Deterministic short-circuit for "retrieval genuinely found nothing" —
    # added after a real, observed failure: the LLM was called with zero
    # context anyway and fabricated both an answer and a citation to a
    # filename that doesn't exist in the corpus, ignoring its own "say you
    # don't have it" system-prompt instruction outright. Don't rely on
    # prompt-following for something this consequential when a
    # deterministic check costs nothing and also saves an LLM call.
    #
    # Branches on is_existence_check (Phase 6 follow-up) — a yes/no
    # question about document content ("did X use ResNet") should get a
    # plain "No, not mentioned" on zero hits, not the web-search offer,
    # which doesn't make sense for a question specifically about this
    # document's own content. Everything else keeps the offer, echoing
    # the query so a later "yes" gives planner.py enough context (via
    # chat history) to know what to search for.
    if not documents:
        if is_existence_check:
            logger.info("No documents retrieved for an existence-check query — deterministic negative answer.")
            return {
                "messages": [{"role": "assistant", "content": _existence_check_negative()}],
                "backend_used": "none",
                "model_used": "",
            }
        logger.info("No documents retrieved — skipping LLM call, offering web search instead.")
        return {
            "messages": [{"role": "assistant", "content": _offer_web_search(query)}],
            "backend_used": "none",
            "model_used": "",
        }

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
    model_used = getattr(llm_client, "last_model_used", "") or routed_model or ""

    # Second refusal path (Phase 6 follow-up) — retrieval found SOME hits
    # (so the zero-hit short-circuit above never fired), but they were
    # irrelevant enough that the LLM followed its own "say you don't have
    # it" instruction anyway. Live-observed: a weather question against an
    # ML-paper corpus got vector_hits=10/kept=5 (nothing zero — hybrid
    # search always returns its top-K, relevant or not), so this refusal
    # was real LLM output, not the deterministic guard, and the web-search
    # offer never fired for it. Normalizing both paths to the same offer
    # here closes that gap without needing a relevance-score threshold
    # (which would be a real feature, not a quick fix — flagged as an
    # Open Problem instead of guessed at here). Same is_existence_check
    # branch applies here as the zero-hit case above, for the same reason.
    if is_refusal(answer):
        logger.info("Responder's own answer was a refusal — applying the same existence-check branch.")
        content = _existence_check_negative() if is_existence_check else _offer_web_search(query)
        return {
            "messages": [{"role": "assistant", "content": content}],
            "backend_used": "none",
            "model_used": "",
        }

    backend_used = getattr(llm_client, "last_backend_used", "") or llm_client.__class__.__name__
    return {
        "messages": [{"role": "assistant", "content": answer}],
        "backend_used": backend_used,
        "model_used": model_used,
    }


async def _answer_conversationally(state: AgentState, query: str) -> dict:
    # Reached by: direct_answer (greetings/chitchat, no history needed)
    # and tool_use for web_search specifically (sandbox has its own
    # deterministic echo branch in generate_node — see there for why).
    # For web_search, tool.py already appended a "Tool Output: ..."
    # message to state["messages"] before this runs (agents/graph.py's
    # edge tool -> responder) — pass recent history so the model can
    # actually synthesize that output, rather than just the bare query.
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
                    "that honestly and specifically (e.g. 'the code sandbox "
                    "couldn't be reached right now' or 'that timed out') — "
                    "do NOT reinterpret an infrastructure failure as a "
                    "capability you lack; you do have these tools, they just "
                    "didn't succeed this time."
                )
            },
            *[{"role": m.get("role", "user"), "content": m.get("content", "")} for m in recent_history],
            {"role": "user", "content": query},
        ],
        temperature=0.5,
        model=routed_model,
    )

    backend_used = getattr(llm_client, "last_backend_used", "") or llm_client.__class__.__name__
    model_used = getattr(llm_client, "last_model_used", "") or routed_model or ""
    return {
        "messages": [{"role": "assistant", "content": answer}],
        "backend_used": backend_used,
        "model_used": model_used,
    }
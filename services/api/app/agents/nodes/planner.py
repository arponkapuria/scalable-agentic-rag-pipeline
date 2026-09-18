import json
import logging
from services.api.app.agents.state import AgentState
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
   "try a weather app"), which is NOT the intended behavior. Bug found
   live: "how's the weather in Dhaka" was misrouted to direct_answer,
   producing exactly that kind of improvised non-answer.

   For "retrieve" specifically, also set "is_existence_check": true if
   the question is a yes/no question asking whether the document(s)
   mention, describe, use, or contain something (e.g. "did the paper use
   ResNet", "is X mentioned", "does the author discuss Y") — as opposed
   to an open-ended informational question ("what does the paper say
   about X", "how does X work"). This matters because if retrieval finds
   nothing, an existence-check question should get a plain "No, that's
   not mentioned" — not the "I don't have that information, want me to
   search the web?" offer, which doesn't make sense for a question that
   is specifically about what THIS document contains. Default to false
   if uncertain — false only means "offer web search on zero hits" (the
   safe default), not "this is never wrong to ask."
3. If the user asks for code execution, output "tool_use" with tool_choice="sandbox".
4. Web search is NEVER triggered on your own initiative. It is only used
   when: the MOST RECENT assistant message in the conversation history
   explicitly asked the user whether to search the web for something
   (this happens automatically when retrieval finds nothing — you do not
   decide when to offer it, only when to act on a confirmed offer), AND
   the user's current message is a clear affirmative response to that
   specific offer (e.g. "yes", "please do", "go ahead", "search it").
   In that case ONLY, output "tool_use" with tool_choice="web_search" and
   tool_input set to the actual topic being searched for — read it from
   the assistant's offer message, which always states it explicitly
   (e.g. an offer like 'I don't have information about "X" in my
   documents. Would you like me to search the web for it?' means
   tool_input should be "X"). If the user's message is NOT a clear
   affirmative to a web-search offer, do not use tool_use for web_search
   under any circumstances — fall back to rule 1 or 2 instead.

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
    """
    Decides the path through the LangGraph.
    """
    logger.info("Planner Node: Analyzing query...")

    # Use current_query, NOT messages[-1] — current_query is the query
    # rewriter's already coreference-resolved, standalone version (e.g.
    # "you answered only half of the question" -> "How does the
    # Transformer improve upon LSTM performance and training efficiency?").
    # messages[-1] deliberately stays the raw user message (preserves real
    # conversation history for memory/logging), which is correct for that
    # purpose but wrong as this node's own input — using it here fed the
    # planner meta-comments/pronouns with zero history in its own LLM
    # call, producing a content-free refined_query. Bug found live: this
    # exact path returned refined_query="complete answer to the user's
    # question" for a "you only answered half" follow-up, and retrieval
    # ran on that literal nonsense string instead of the real question.
    user_query = state.get("current_query") or ""
    if not user_query:
        last_message = state["messages"][-1]
        user_query = last_message.get("content", "") if isinstance(last_message, dict) else getattr(last_message, "content", "")

    # Last 2 turns of REAL conversation history (Phase 6 follow-up) — the
    # ONLY reason this is here is rule 4 above: detecting "yes" as
    # confirmation of a web-search offer requires actually seeing the
    # previous assistant turn. Deliberately small (not the full history)
    # to keep this call cheap and its behavior easy to reason about —
    # this is not a general-purpose history-aware planner, just enough
    # context for the one specific confirmation case.
    recent_history = (state.get("messages") or [])[-2:]
    history_messages = [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        for m in recent_history if isinstance(m, dict)
    ]

    # Call LLM to plan
    try:
        response_text = await llm_client.chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *history_messages,
                {"role": "user", "content": user_query}
            ],
            temperature=0.0, # Deterministic planning
            json_mode=True,
            # Reuses model_router.py's EXISTING complexity heuristic
            # (already used for responder's synthesis call) instead of a
            # blanket MODEL_TIER_SIMPLE pin — same rule, applied
            # consistently everywhere model-tier choice matters, rather
            # than a separate ad-hoc rule per call site. Grounded in a
            # live-observed failure: MODEL_TIER_SIMPLE (a small/fast
            # model) returned syntactically valid JSON but misclassified
            # a "how is X different from Y and how does it improve from
            # Y" compound question as direct_answer — exactly the kind of
            # phrasing model_router.py's own COMPLEX_KEYWORDS list
            # already flags ("difference between") for the bigger tier.
            # Falls through to the rest of Groq's model list on failure
            # (see openai_compatible.py) rather than hard-failing.
            model=route_model(user_query),
        )
        
        # Parse JSON
        plan = json.loads(response_text)
        
        logger.info(f"Plan derived: {plan['action']}")
        
        # Update State
        # `plan.get(key, default)` only falls back when the KEY is
        # missing — not when the LLM's JSON has the key present with an
        # explicit `null` value, which is exactly what a reasonable
        # response looks like for "refined_query"/"tool_choice"/
        # "tool_input" on a direct_answer/no-tool turn (e.g. a plain
        # "hey" greeting). `.get()` alone let a null refined_query
        # silently overwrite current_query with None, crashing
        # route_model() downstream with 'NoneType' has no attribute
        # 'lower' — found live. `or` correctly falls back on both
        # missing-key AND explicit-null cases.
        return {
            "current_query": plan.get("refined_query") or user_query,
            "plan": [plan["reasoning"]],
            "action": plan.get("action") or "retrieve",
            "tool_choice": plan.get("tool_choice") or "",
            "tool_input": plan.get("tool_input") or "",
            "is_existence_check": bool(plan.get("is_existence_check")),
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
            "action": "retrieve",
            "is_existence_check": False,  # safe default: offer web search on zero hits
        }
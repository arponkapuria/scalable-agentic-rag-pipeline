"""
POST /stream
      ↓
Check L1 exact cache (current corpus_version) → hit? stream instantly
      ↓ miss
Check L2 semantic cache (RAG-sourced answers only) → hit? stream instantly
      ↓ miss
Load conversation history from PostgreSQL
      ↓
[Optional] Rewrite query to resolve coreferences
      ↓
[Optional] Generate hypothetical document (HyDE) for better retrieval
      ↓
Initialize LangGraph agent state
      ↓
Stream agent events (planner → retriever/tool → responder)
      ↓
Save to memory (before yielding, so message_id is available for
feedback linking) + write cache (L1 for vector_search/direct_answer
results, skipped for web_search/sandbox and skipped if the answer is a
known refusal) + before/after comparison if an older-version entry existed
"""

import logging
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from services.api.app.session.dependency import get_corpus_id
from services.api.app.cache.redis_cache import RedisCache, redis_cache as global_cache
from services.api.app.memory.postgres import PostgresMemory, postgres_memory as global_memory
from services.api.app.clients.llm.factory import llm_client as global_llm
from services.api.app.clients.llm.base import LLMClient
from services.api.app.agents.graph import agent_app
from services.api.app.agents.state import AgentState
from services.api.app.enhancers.query_rewriter import rewrite_query
from services.api.app.enhancers.hyde import generate_hypothetical_document
from libs.utils.refusal import is_refusal as _is_refusal

router = APIRouter()
logger = logging.getLogger(__name__)

# Never cached (L1), regardless of answer content (Phase 6 follow-up) —
# web_search results go stale over time; sandbox output isn't meant to
# be replayed as a cached "answer" either. Overrides
# PROJECT_INSTRUCTIONS.md's original "L1 applies to every tool's answer"
# line for these two tools specifically — deliberate, not an oversight
# if you're diffing against that doc.
NO_CACHE_TOOLS = ("web_search")

# --- Dependency Providers (DI) ---

def get_cache() -> RedisCache:
    return global_cache

def get_memory() -> PostgresMemory:
    return global_memory

def get_llm_client() -> LLMClient:
    return global_llm

# --- Schemas ---

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's query")

    # Query rewriter defaults ON — coreference resolution is cheap
    # (one small LLM call) and strictly helps multi-turn retrieval.
    # HyDE stays default-off per the locked design (not evaluated yet).
    use_query_rewriter: bool = Field(default=True, description="Resolve coreferences using conversation history")
    use_hyde: bool = Field(default=False, description="Generate hypothetical document for better retrieval")


class PreviousAnswer(BaseModel):
    content: str
    corpus_version: int
    cached_at: str


class ChatResponse(BaseModel):
    """Formalized (Phase 6) — previously hand-built dicts at each of the
    four places a response is emitted (exact-hit, semantic-hit, live
    answer, error). Same wire shape as before, now a single typed
    definition the frontend contract can be checked against. Each NDJSON
    line is one of these serialized, except "status" events (see
    StatusEvent below), which are a distinct, lighter-weight shape."""
    type: str = "answer"  # "answer" | "error"
    content: str
    corpus_id: Optional[str] = None
    # ChatHistory row id for the ASSISTANT message this response
    # represents (Phase 6 follow-up) — lets the frontend attach feedback
    # to a specific response via POST /feedback's message_id. None only
    # on the "error" type, where nothing was persisted.
    message_id: Optional[int] = None
    cache_hit: str = "none"  # "none" | "exact" | "semantic"
    tool_used: str = ""
    backend_used: str = ""
    model_used: str = ""  # actual model id, e.g. "openai/gpt-oss-120b" — "" when no LLM call was made (Phase 6 follow-up)
    sources: list[str] = Field(default_factory=list)
    previous_answer: Optional[PreviousAnswer] = None


class StatusEvent(BaseModel):
    """One per LangGraph node completion — lets the frontend show live
    progress (e.g. "Retrieving documents…") instead of a silent wait
    while the NDJSON stream is open, the same way ChatGPT/Claude show
    step indicators during a response."""
    type: str = "status"
    node: str
    corpus_id: str
    info: str

# --- Routes ---

def _cache_hit_payload(corpus_id: str, entry: dict, cache_hit: str, message_id: int, older_entry: dict | None = None) -> str:
    previous = PreviousAnswer(
        content=older_entry["answer"],
        corpus_version=older_entry["corpus_version"],
        cached_at=older_entry["cached_at"],
    ) if older_entry else None
    # Locked design: intentional demo feature, not a staleness workaround —
    # surface both, don't just silently prefer the newer one.
    response = ChatResponse(
        content=entry["answer"],
        corpus_id=corpus_id,
        message_id=message_id,
        cache_hit=cache_hit,  # "exact" | "semantic"
        tool_used=entry.get("tool_used", ""),
        backend_used=entry.get("backend_used", ""),
        model_used=entry.get("model_used", ""),
        sources=entry.get("sources", []),
        previous_answer=previous,
    )
    return response.model_dump_json() + "\n"


@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    corpus_id: str = Depends(get_corpus_id),
    cache: RedisCache = Depends(get_cache),
    memory: PostgresMemory = Depends(get_memory),
    llm: LLMClient = Depends(get_llm_client)  # unused below — kept for DI/testability; nodes import the global singleton directly
):
    """
    Main Chat Endpoint (Streaming).
    Orchestrates the RAG flow: Cache -> Enhance -> History -> Agent -> Stream -> Cache-write.
    """
    logger.info(f"Chat request for corpus {corpus_id}")

    # 1. L1 exact-match (current corpus_version only)
    exact_hit = await cache.get_exact(corpus_id, req.message)
    # Defensive skip for entries cached before the refusal-gate fix below
    # existed — don't serve a stored refusal as if it were a real cached
    # answer; fall through to a genuine pipeline run instead.
    if exact_hit and _is_refusal(exact_hit["answer"]):
        logger.info(f"L1 exact hit was a stored refusal (pre-fix entry) — treating as miss")
        exact_hit = None
    if exact_hit:
        logger.info(f"L1 exact cache hit for corpus {corpus_id}")

        async def stream_exact():
            await memory.add_message(corpus_id, "user", req.message)
            assistant_id = await memory.add_message(corpus_id, "assistant", exact_hit["answer"])
            yield _cache_hit_payload(corpus_id, exact_hit, "exact", assistant_id)

        return StreamingResponse(stream_exact(), media_type="application/x-ndjson")

    # 2. L2 semantic-match (RAG-sourced answers only — see redis_cache.py)
    semantic_hit = await cache.get_semantic(corpus_id, req.message)
    if semantic_hit and _is_refusal(semantic_hit["answer"]):
        logger.info(f"L2 semantic hit was a stored refusal (pre-fix entry) — treating as miss")
        semantic_hit = None
    if semantic_hit:
        logger.info(f"L2 semantic cache hit for corpus {corpus_id}")

        async def stream_semantic():
            await memory.add_message(corpus_id, "user", req.message)
            assistant_id = await memory.add_message(corpus_id, "assistant", semantic_hit["answer"])
            yield _cache_hit_payload(corpus_id, semantic_hit, "semantic", assistant_id)

        return StreamingResponse(stream_semantic(), media_type="application/x-ndjson")

    # No current-version hit — check for a STALE entry (older corpus_version)
    # now, before running the pipeline, so we can offer the before/after
    # comparison once the fresh answer is ready (locked design: run fresh,
    # cache the new result, return BOTH).
    older_versions = await cache.get_all_versions(corpus_id, req.message)
    stale_entry = older_versions[0] if older_versions else None

    # 3. Load Conversation History
    history_objs = await memory.get_history(corpus_id, limit=6)
    history_dicts = [
        {"role": msg.role, "content": msg.content} for msg in history_objs
    ]

    # 4. Query Enhancement (Optional)
    # Start with the raw user message
    enhanced_query = req.message

    # Step A — Query Rewriter: resolve coreferences using history
    # "How much does it cost?" + history → "How much does Kubernetes cost?"
    if req.use_query_rewriter and history_dicts:
        try:
            enhanced_query = await rewrite_query(enhanced_query, history_dicts)
            logger.info(f"Query rewritten: '{req.message}' → '{enhanced_query}'")
        except Exception as e:
            # Non-fatal — fall back to original query
            logger.warning(f"Query rewriter failed, using original: {e}")
            enhanced_query = req.message

    # Step B — HyDE: generate hypothetical document for better vector similarity
    # "What is Kubernetes?" → fake paragraph using K8s vocabulary → better embedding match
    if req.use_hyde:
        try:
            enhanced_query = await generate_hypothetical_document(enhanced_query)
            logger.info(f"HyDE query generated for: '{req.message}'")
        except Exception as e:
            # Non-fatal — fall back to rewritten or original query
            logger.warning(f"HyDE generation failed, using previous query: {e}")

    # 5. Append current user message to history
    history_dicts.append({"role": "user", "content": req.message})

    # 6. Initialize Agent State
    # Note: current_query uses the enhanced version for better retrieval
    #       messages uses the original message to preserve conversation history correctly
    initial_state = AgentState(
        messages=history_dicts,
        current_query=enhanced_query,   # ← enhanced query for retrieval
        documents=[],
        plan=[],
        action="",
        corpus_id=corpus_id,
        tool_used="",
        backend_used="",
        model_used="",
        sources=[],
        is_existence_check=False,
    )

    # 7. Streaming Generator
    async def event_generator() -> AsyncGenerator[str, None]:
        final_answer = ""
        final_tool_used = ""
        final_backend_used = ""
        final_model_used = ""
        final_sources: list[str] = []

        try:
            async for event in agent_app.astream(initial_state):
                node_name = list(event.keys())[0]
                node_data = event[node_name]

                # Emit status for every node so frontend can show progress
                yield StatusEvent(
                    node=node_name,
                    corpus_id=corpus_id,
                    info=f"Completed step: {node_name}",
                ).model_dump_json() + "\n"

                if "tool_used" in node_data and node_data["tool_used"]:
                    final_tool_used = node_data["tool_used"]
                if "backend_used" in node_data and node_data["backend_used"]:
                    final_backend_used = node_data["backend_used"]
                if "sources" in node_data and node_data["sources"]:
                    final_sources = node_data["sources"]

                # Capture final answer from responder node
                if node_name == "responder":
                    if "messages" in node_data and node_data["messages"]:
                        ai_msg = node_data["messages"][-1]
                        final_answer = ai_msg.get("content", "")
                        # Only ever set by responder — captured directly here
                        # rather than the generic per-node block above, since
                        # an empty string is a legitimate value (no LLM call
                        # made, e.g. sandbox echo/deterministic guards) and
                        # the generic block's truthy check would incorrectly
                        # skip updating it in that case.
                        final_model_used = node_data.get("model_used", "")

                        # Persisted here, before yielding, so the response
                        # can carry the assistant message's real id for
                        # feedback linking (Phase 6 follow-up) — doing
                        # this in post-processing (after the yield) would
                        # be too late for the client to have it.
                        await memory.add_message(corpus_id, "user", req.message)
                        assistant_id = await memory.add_message(corpus_id, "assistant", final_answer)

                        previous = PreviousAnswer(
                            content=stale_entry["answer"],
                            corpus_version=stale_entry["corpus_version"],
                            cached_at=stale_entry["cached_at"],
                        ) if stale_entry else None

                        response = ChatResponse(
                            content=final_answer,
                            corpus_id=corpus_id,
                            message_id=assistant_id,
                            cache_hit="none",
                            tool_used=final_tool_used,
                            backend_used=final_backend_used,
                            model_used=final_model_used,
                            sources=final_sources,
                            previous_answer=previous,
                        )
                        yield response.model_dump_json() + "\n"

            # 8. Post-Processing — cache write only; conversation history
            # was already persisted above (needed the id before yielding).
            # Gated on _is_refusal (see REFUSAL_PREFIXES's docstring in
            # libs/utils/refusal.py) and NO_CACHE_TOOLS (see its
            # module-level comment above).
            if final_answer and not _is_refusal(final_answer) and final_tool_used not in NO_CACHE_TOOLS:
                await cache.set_exact(
                    corpus_id, req.message, final_answer, final_sources,
                    final_tool_used, final_backend_used, final_model_used,
                )
            elif final_answer and final_tool_used in NO_CACHE_TOOLS:
                logger.info(f"tool_used={final_tool_used} — not caching")
            elif final_answer:
                logger.info("Refusal answer — not caching")

        except Exception as e:
            logger.error(f"Error in chat stream: {e}", exc_info=True)
            yield ChatResponse(type="error", content="An internal error occurred.").model_dump_json() + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

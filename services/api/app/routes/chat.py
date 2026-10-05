"""
Main chat endpoint. Checks L1 exact cache, then L2 semantic cache, then runs query enhancement and the LangGraph agent (planner -> retriever/tool -> responder), streaming status events. Persists the exchange and writes the cache entry, including a before/after comparison when a stale cached entry exists.
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
from libs.utils.refusal import GUARDRAIL_INPUT_BLOCKED, is_refusal as _is_refusal, is_truncated as _is_truncated
from libs.guardrails.filters import check_input

router = APIRouter()
logger = logging.getLogger(__name__)

NO_CACHE_TOOLS = ("web_search")  # never cached — web results go stale, sandbox output isn't a replayable answer


def get_cache() -> RedisCache:
    return global_cache

def get_memory() -> PostgresMemory:
    return global_memory

def get_llm_client() -> LLMClient:
    return global_llm


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's query")
    use_query_rewriter: bool = Field(default=True, description="Resolve coreferences using conversation history")
    use_hyde: bool = Field(default=False, description="Generate hypothetical document for better retrieval")


class PreviousAnswer(BaseModel):
    content: str
    corpus_version: int
    cached_at: str


class ChatResponse(BaseModel):
    """One response to a chat turn, serialized as one NDJSON line (see StatusEvent for the other line type)."""
    type: str = "answer"  # "answer" | "error"
    content: str
    corpus_id: Optional[str] = None
    message_id: Optional[int] = None  # ChatHistory row id, for feedback linking
    cache_hit: str = "none"  # "none" | "exact" | "semantic"
    tool_used: str = ""
    backend_used: str = ""
    model_used: str = ""
    sources: list[str] = Field(default_factory=list)
    previous_answer: Optional[PreviousAnswer] = None


class StatusEvent(BaseModel):
    """One per LangGraph node completion, for the frontend's live progress indicator."""
    type: str = "status"
    node: str
    corpus_id: str
    info: str


def _cache_hit_payload(corpus_id: str, entry: dict, cache_hit: str, message_id: int, older_entry: dict | None = None) -> str:
    """Builds the NDJSON line for a cache hit, including a stale-entry comparison if one exists.

    Args:
        corpus_id: The caller's session.
        entry: The cache entry being served.
        cache_hit: "exact" or "semantic".
        message_id: The persisted assistant message's row id.
        older_entry: A stale cache entry at an older corpus_version, if one exists.

    Returns:
        One NDJSON line.
    """
    previous = PreviousAnswer(
        content=older_entry["answer"],
        corpus_version=older_entry["corpus_version"],
        cached_at=older_entry["cached_at"],
    ) if older_entry else None
    response = ChatResponse(
        content=entry["answer"],
        corpus_id=corpus_id,
        message_id=message_id,
        cache_hit=cache_hit,
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
    llm: LLMClient = Depends(get_llm_client)  # unused — kept for DI/testability
):
    """Streams one chat turn's response as NDJSON: cache check, query enhancement, agent run, cache write.

    Args:
        req: The chat request.
        corpus_id: The caller's session, injected from the session cookie.
        cache: The response cache.
        memory: The conversation history store.
        llm: Unused directly — present for dependency injection/testability.

    Returns:
        A StreamingResponse of NDJSON lines.
    """
    logger.info(f"Chat request for corpus {corpus_id}")

    blocked_reason = check_input(req.message)
    if blocked_reason:
        logger.info(f"Input guardrail blocked request ({blocked_reason})")

        async def stream_blocked():
            await memory.add_message(corpus_id, "user", req.message)
            assistant_id = await memory.add_message(corpus_id, "assistant", GUARDRAIL_INPUT_BLOCKED)
            yield ChatResponse(
                content=GUARDRAIL_INPUT_BLOCKED, corpus_id=corpus_id, message_id=assistant_id,
                cache_hit="none", tool_used="", backend_used="none", model_used="", sources=[],
            ).model_dump_json() + "\n"

        return StreamingResponse(stream_blocked(), media_type="application/x-ndjson")

    # L1 exact match
    exact_hit = await cache.get_exact(corpus_id, req.message)
    if exact_hit and _is_refusal(exact_hit["answer"]):
        exact_hit = None  # pre-fix cached refusal — treat as a miss
    if exact_hit:
        logger.info(f"L1 exact cache hit for corpus {corpus_id}")

        async def stream_exact():
            await memory.add_message(corpus_id, "user", req.message)
            assistant_id = await memory.add_message(corpus_id, "assistant", exact_hit["answer"])
            yield _cache_hit_payload(corpus_id, exact_hit, "exact", assistant_id)

        return StreamingResponse(stream_exact(), media_type="application/x-ndjson")

    # L2 semantic match (vector_search-sourced answers only)
    semantic_hit = await cache.get_semantic(corpus_id, req.message)
    if semantic_hit and _is_refusal(semantic_hit["answer"]):
        semantic_hit = None
    if semantic_hit:
        logger.info(f"L2 semantic cache hit for corpus {corpus_id}")

        async def stream_semantic():
            await memory.add_message(corpus_id, "user", req.message)
            assistant_id = await memory.add_message(corpus_id, "assistant", semantic_hit["answer"])
            yield _cache_hit_payload(corpus_id, semantic_hit, "semantic", assistant_id)

        return StreamingResponse(stream_semantic(), media_type="application/x-ndjson")

    # Stale entry, checked before the pipeline runs, for the before/after comparison
    older_versions = await cache.get_all_versions(corpus_id, req.message)
    stale_entry = older_versions[0] if older_versions else None

    history_objs = await memory.get_history(corpus_id, limit=6)
    history_dicts = [{"role": msg.role, "content": msg.content} for msg in history_objs]

    enhanced_query = req.message

    if req.use_query_rewriter and history_dicts:
        try:
            enhanced_query = await rewrite_query(enhanced_query, history_dicts)
            logger.info(f"Query rewritten: '{req.message}' -> '{enhanced_query}'")
        except Exception as e:
            logger.warning(f"Query rewriter failed, using original: {e}")
            enhanced_query = req.message

    if req.use_hyde:
        try:
            enhanced_query = await generate_hypothetical_document(enhanced_query)
            logger.info(f"HyDE query generated for: '{req.message}'")
        except Exception as e:
            logger.warning(f"HyDE generation failed, using previous query: {e}")

    history_dicts.append({"role": "user", "content": req.message})

    initial_state = AgentState(
        messages=history_dicts,
        current_query=enhanced_query,
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

                if node_name == "responder":
                    if "messages" in node_data and node_data["messages"]:
                        ai_msg = node_data["messages"][-1]
                        final_answer = ai_msg.get("content", "")
                        final_model_used = node_data.get("model_used", "")

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

            if final_answer and not _is_refusal(final_answer) and not _is_truncated(final_answer) and final_tool_used not in NO_CACHE_TOOLS:
                await cache.set_exact(
                    corpus_id, req.message, final_answer, final_sources,
                    final_tool_used, final_backend_used, final_model_used,
                )
            elif final_answer:
                logger.info("Not caching — non-cacheable tool, refusal, or truncated answer.")

        except Exception as e:
            logger.error(f"Error in chat stream: {e}", exc_info=True)
            yield ChatResponse(type="error", content="An internal error occurred.").model_dump_json() + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")
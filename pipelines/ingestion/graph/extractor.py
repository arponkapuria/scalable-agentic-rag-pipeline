"""
Ray worker that sends text chunks to an LLM to extract knowledge-graph
entities/relationships, returning nodes/edges per chunk.

Batches multiple chunks into ONE LLM call (Bug 2 fix) instead of one call
per chunk — on a 122-chunk paper this was 122 sequential Groq calls,
producing 494 429s / 110 full-backend exhaustions in testing. Batching by
Ray's own batch_size (main.py sets batch_size=5 for this stage) cuts that
to ~1 call per 5 chunks with no separate config needed.

The LLM client (and its circuit breaker) is created ONCE per actor, in
__init__, and reused across every batch __call__ for that actor's whole
lifetime — NOT recreated per batch. This was a real bug in the first cut
of this file: a fresh client per batch meant a fresh (CLOSED) circuit
breaker every time too, so the "stop hammering during cooldown" mechanism
never actually engaged — every batch retried the full Groq->OpenRouter
sequence from scratch even while Groq was still fully exhausted,
producing a 27-minute ingestion run dominated by retry/backoff waits, not
real work. A persistent event loop (not asyncio.run() per call) is what
makes a persistent httpx.AsyncClient safe to reuse here — Ray actors are
long-lived processes, so one loop for the actor's life is the correct
pattern, not a new one per batch.

Rate-limiter-sharing bug (fixed): with Ray's ActorPoolStrategy running 2
concurrent GraphExtractor actors, each built its OWN LLMClient via
build_llm_client() — two separate BackendRateLimiter/CircuitBreaker
instances undercounting one real, shared Groq/OpenRouter account budget.
`__init__` now takes an optional `llm_client` — the in-process ingestion
path (pipelines/ingestion/pipeline.py) passes in the FastAPI app's single
global `llm_client` singleton, the same one chat uses, so there's only
ever ONE rate limiter/circuit breaker for the whole process, chat and
ingestion included (correct: the provider's limit is account-level
regardless of which logical component is calling). When llm_client is
omitted (the Ray-actor `__call__` path, still separate processes), this
class builds and owns its own client exactly as before — the sharing fix
only applies where actual process-sharing is possible.

Output nodes/edges are JSON-STRING serialized, not native Python lists
(Bug 4 fix) — Ray Data batches are Arrow-backed, and a column of
ragged/mixed-shape object arrays (some rows [], others lists-of-dicts)
broke Arrow's type inference, silently falling back to slow pickled
serialization every batch. A plain string column has no such problem;
Neo4jIndexer json.loads()s it back.

Uses the Phase 1 LLMClient factory (Groq/OpenRouter failover, matching the
chat-response path) — not the dead Ray Serve LLM endpoint this originally
pointed at.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from pipelines.ingestion.graph.schema import GraphSchema
from services.api.app.clients.llm.base import LLMClient
from services.api.app.clients.llm.factory import build_llm_client
from services.api.app.clients.llm.openai_compatible import ModelExhaustedError

logger = logging.getLogger(__name__)


class GraphExtractor:
    """Callable class used by both the Ray Data path (__call__) and the
    in-process path (await extract_batch(...) directly, no Ray)."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        # A loop is always created for __call__'s sake (Ray Data invokes
        # this synchronously) — but if llm_client is a shared instance
        # from the app's own running loop, __call__ shouldn't be mixed
        # with in-process/await usage on the same extractor instance;
        # the two entry points are for the two different execution
        # backends (INGESTION_BACKEND=ray vs in_process), not both at once.
        self._loop = asyncio.new_event_loop()
        self._owns_client = llm_client is None
        if llm_client is not None:
            self._client = llm_client  # already started by app lifespan
        else:
            self._client = build_llm_client()
            self._loop.run_until_complete(self._client.start())

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        texts = list(batch["text"])  # Ray batches are numpy arrays by default — see compute.py's same fix
        results = self._loop.run_until_complete(self.extract_batch(texts))

        batch["graph_nodes"] = [json.dumps(nodes) for nodes, _ in results]
        batch["graph_edges"] = [json.dumps(edges) for _, edges in results]
        return batch

    async def extract_batch(self, texts: List[str]) -> List[Tuple[list, list]]:
        numbered = "\n\n".join(f"[Segment {i}]\n{text}" for i, text in enumerate(texts))
        try:
            content = await self._client.chat_completion(
                messages=[
                    {"role": "system", "content": GraphSchema.get_batch_system_prompt()},
                    {"role": "user", "content": numbered},
                ],
                temperature=0.0,
                json_mode=True,
                # The default 1024 was sized for a single chunk's output,
                # not N segments' worth of nodes+edges JSON — too small
                # here truncates mid-string, which is what produced the
                # "Unterminated string"/segment-count-mismatch failures.
                # Capped at 950, not scaled up freely: live testing showed
                # qwen/qwen3.8-27b hard-rejects any single request asking
                # for >=1000 max_tokens (its real OTPM ceiling), so this
                # stays safely under that regardless of batch size —
                # GRAPH_EXTRACTION_BATCH_SIZE is what should be tuned to
                # fit within this cap, not the other way around.
                max_tokens=min(950, 350 * len(texts)),
            )
            segments = json.loads(content).get("segments", [])
            if len(segments) != len(texts):
                raise ValueError(
                    f"batch extraction returned {len(segments)} segments for {len(texts)} inputs"
                )
            return [(seg.get("nodes", []), seg.get("edges", [])) for seg in segments]

        except ModelExhaustedError as e:
            # The batch call already tried every model (with its own
            # retries) and every one failed/rate-limited — that's the
            # WHOLE backend exhausted, not a parsing fluke. Falling back
            # to N individual calls here would just rediscover the same
            # exhaustion N times, burning scarce quota at exactly the
            # moment it's scarcest (this was the actual cause of runs
            # stretching to 20+ minutes under load). Skip the batch
            # instead — ingestion still completes, just with no graph
            # data for these chunks, which is a better tradeoff than a
            # retry storm.
            logger.warning(f"Batch graph extraction skipped ({len(texts)} chunks) — backend exhausted: {e}")
            return [([], []) for _ in texts]

        except Exception as e:
            # Malformed/misaligned JSON, not exhaustion — worth retrying
            # per chunk since a single chunk's smaller payload is less
            # likely to hit the same truncation/parsing issue.
            logger.warning(f"Batch graph extraction failed ({e}); falling back to per-chunk calls")
            results = []
            for text in texts:
                try:
                    results.append(await self._extract_one(text))
                except ModelExhaustedError as exhausted:
                    # Backend went from "some models slow" to "fully
                    # exhausted" partway through the fallback loop — stop
                    # spending remaining chunks' worth of calls on a
                    # backend that's already down; skip the rest of this
                    # batch too instead of retrying each one individually.
                    logger.warning(f"Per-chunk fallback aborted, backend exhausted: {exhausted}")
                    remaining = len(texts) - len(results)
                    results.extend([([], [])] * remaining)
                    break
            return results

    async def _extract_one(self, text: str) -> Tuple[list, list]:
        try:
            content = await self._client.chat_completion(
                messages=[
                    {"role": "system", "content": GraphSchema.get_system_prompt()},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                json_mode=True,
            )
            graph_data = json.loads(content)
            return graph_data.get("nodes", []), graph_data.get("edges", [])
        except ModelExhaustedError:
            # Let this propagate — the caller (extract_batch's fallback
            # loop) needs to see it to stop calling _extract_one for the
            # rest of this batch, instead of each subsequent call
            # independently rediscovering the same exhaustion.
            raise
        except Exception as e:
            logger.warning(f"Graph extraction failed for chunk: {e}")
            return [], []

    def __del__(self):
        # Best-effort cleanup when the actor/instance is torn down — not
        # critical (the process is exiting anyway), just avoids a
        # dangling unclosed-client warning in logs. Never closes a
        # shared (not owned) client — that's the app lifespan's job.
        try:
            if self._owns_client:
                self._loop.run_until_complete(self._client.close())
            self._loop.close()
        except Exception:
            pass
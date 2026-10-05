"""
FastAPI application entry point: wires every route, manages startup/shutdown of all client connections, and serves the static chat UI.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import services.api.app.logging  # noqa: F401 — configures the root logger; must import first
from services.api.app.clients.qdrant import qdrant_client
from services.api.app.clients.llm.factory import llm_client
from services.api.app.clients.llm.mistral_client import mistral_client
from services.api.app.cache.redis import redis_client
from services.api.app.cache.redis_cache import redis_cache
from services.api.app.memory.models import Base
from services.api.app.memory.postgres import engine, document_store
from services.api.app.session.cleanup import start_cleanup_task, stop_cleanup_task
from services.api.app.routes import chat, upload, health, feedback, session, webhooks, ingest_status, corpus


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages startup and shutdown of every connection pool and background task.

    Args:
        app: The FastAPI application instance.
    """

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # A document row still non-terminal at boot belonged to a now-dead process — mark it failed
    # instead of leaving it stuck forever.
    orphaned = await document_store.fail_orphaned()
    if orphaned:
        print(f"Marked {orphaned} orphaned document(s) as failed (interrupted by a previous restart).")

    print("Initializing clients...")
    await redis_client.connect()
    await llm_client.start()
    await mistral_client.start()

    # qdrant_client.init_collections() is lazy — connects on first real use, not here, so the
    # app stays bootable without a vector-search profile up.

    cleanup_task = start_cleanup_task()

    yield

    print("Closing clients...")
    await stop_cleanup_task(cleanup_task)
    await redis_client.close()
    await llm_client.close()
    await mistral_client.close()
    await qdrant_client.close()
    await redis_cache.close()


app = FastAPI(title="DocRAG - Scalable Agentic RAG Pipeline", version="1.0.0", lifespan=lifespan)

app.include_router(session.router, prefix="/api/v1/session", tags=["Session"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(upload.router, prefix="/api/v1/upload", tags=["Upload"])
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(feedback.router, prefix="/api/v1/feedback", tags=["Feedback"])
app.include_router(webhooks.router, prefix="/api/v1/webhooks", tags=["Webhooks"])
app.include_router(ingest_status.router, prefix="/api/v1/ingest", tags=["Ingest"])
app.include_router(corpus.router, prefix="/api/v1/corpus", tags=["Corpus"])

app.mount("/", StaticFiles(directory="services/api/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
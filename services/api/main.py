from contextlib import asynccontextmanager
from fastapi import FastAPI

import services.api.app.logging  # noqa: F401 — import alone runs setup_logging()
                                  # via the module's own "initialize on
                                  # import" call; must be the first
                                  # first-party import in this file so the
                                  # root logger is configured (rotating
                                  # file + console) before anything else
                                  # calls logging.getLogger(). Previously
                                  # nothing imported this module at all,
                                  # so the root logger stayed at Python's
                                  # default WARNING and every logger.info()
                                  # call across the app was silently
                                  # dropped — uvicorn's own access/error
                                  # loggers are unaffected by this, which
                                  # is why request lines always showed up
                                  # while nothing else did.
from services.api.app.clients.qdrant import qdrant_client
from services.api.app.clients.llm.factory import llm_client
from services.api.app.clients.llm.gemini_client import gemini_client
from services.api.app.cache.redis import redis_client
from services.api.app.cache.redis_cache import redis_cache
from services.api.app.memory.models import Base, ChatHistory, Feedback
from services.api.app.memory.postgres import engine
from services.api.app.session.cleanup import start_cleanup_task, stop_cleanup_task
from services.api.app.routes import chat, upload, health, feedback, session, webhooks

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Centralized Resource Management.
    Initialize all connection pools here.
    """

    # Create all tables defined in models.py
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # 1. Startup
    print("Initializing clients...")
    await redis_client.connect()
    await llm_client.start()
    # Vision-only client, used by in-process ingestion's PDF captioning —
    # started here so it's ready before any upload triggers the webhook.
    await gemini_client.start()

    # NOTE: qdrant_client.init_collections() is intentionally NOT called
    # here — it's lazy now (see clients/qdrant.py), connecting on first
    # real use instead of unconditionally at boot. Keeps the app bootable
    # on whatever Docker profile is currently up.

    cleanup_task = start_cleanup_task()

    yield
    
    # 2. Shutdown
    print("Closing clients...")
    await stop_cleanup_task(cleanup_task)
    await redis_client.close()
    await llm_client.close()
    await gemini_client.close()
    await qdrant_client.close()
    await redis_cache.close()

# FastAPI Application
app = FastAPI(title="OmniRAG - Scalable Agentic RAG Platform", version="1.0.0", lifespan=lifespan)

# Include Routes
app.include_router(session.router, prefix="/api/v1/session", tags=["Session"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(upload.router, prefix="/api/v1/upload", tags=["Upload"])
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(feedback.router, prefix="/api/v1/feedback", tags=["Feedback"])
app.include_router(webhooks.router, prefix="/api/v1/webhooks", tags=["Webhooks"])

if __name__ == "__main__":
    import uvicorn
    # In production, this is run via Gunicorn/Uvicorn in Docker
    uvicorn.run(app, host="0.0.0.0", port=8000)
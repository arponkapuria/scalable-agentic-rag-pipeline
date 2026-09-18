from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, delete
from services.api.app.config import settings
from services.api.app.memory.models import ChatHistory, Feedback, Document

# Async Engine & Session
engine = create_async_engine(settings.DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)

class PostgresMemory:
    """
    Manager for persisting conversation state.
    """
    async def add_message(self, corpus_id: str, role: str, content: str) -> int:
        """Returns the new row's id — needed (Phase 6 follow-up) so the
        frontend can attach feedback to a specific assistant message via
        POST /feedback's message_id field."""
        async with AsyncSessionLocal() as session:
            async with session.begin():
                msg = ChatHistory(
                    corpus_id=corpus_id,
                    role=role,
                    content=content,
                )
                session.add(msg)
                await session.flush()  # populates msg.id before commit
                return msg.id

    async def get_history(self, corpus_id: str, limit: int = 10):
        """
        Fetch last N messages for context window.
        """
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(ChatHistory)
                .where(ChatHistory.corpus_id == corpus_id)
                .order_by(ChatHistory.created_at.desc())
                .limit(limit)
            )
            # Reverse to get chronological order (Oldest -> Newest)
            return result.scalars().all()[::-1]

    async def delete_by_corpus_id(self, corpus_id: str) -> None:
        """Cascade-delete hook for session expiry (session/cleanup.py) —
        removes this corpus's chat history, feedback, and document rows.
        None of the three has a DB-level FK/cascade constraint between
        them, so each is cleared explicitly here rather than relying on
        ON DELETE CASCADE."""
        async with AsyncSessionLocal() as session:
            async with session.begin():
                await session.execute(delete(ChatHistory).where(ChatHistory.corpus_id == corpus_id))
                await session.execute(delete(Feedback).where(Feedback.corpus_id == corpus_id))
                await session.execute(delete(Document).where(Document.corpus_id == corpus_id))


class DocumentStore:
    """CRUD for the 'documents' table — the persisted state ingest/status
    and corpus/documents read from (Phase 6)."""

    async def create_pending(self, file_id: str, corpus_id: str, filename: str, s3_key: str) -> None:
        async with AsyncSessionLocal() as session:
            async with session.begin():
                session.add(Document(
                    file_id=file_id, corpus_id=corpus_id, filename=filename,
                    s3_key=s3_key, status="pending",
                ))

    async def set_status(self, file_id: str, status: str, error: Optional[str] = None) -> None:
        """Called by pipeline.py after each stage. error is only ever set
        alongside status="failed"."""
        async with AsyncSessionLocal() as session:
            async with session.begin():
                result = await session.execute(select(Document).where(Document.file_id == file_id))
                doc = result.scalar_one_or_none()
                if doc is None:
                    return  # shouldn't happen — upload.py always creates the row first
                doc.status = status
                if error is not None:
                    doc.error = error
                if status in ("complete", "failed"):
                    doc.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)

    async def set_complete(self, file_id: str, chunk_count: int) -> None:
        async with AsyncSessionLocal() as session:
            async with session.begin():
                result = await session.execute(select(Document).where(Document.file_id == file_id))
                doc = result.scalar_one_or_none()
                if doc is None:
                    return
                doc.status = "complete"
                doc.chunk_count = chunk_count
                doc.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)

    async def get_by_file_id(self, file_id: str) -> Optional[Document]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(Document).where(Document.file_id == file_id))
            return result.scalar_one_or_none()

    async def list_by_corpus_id(self, corpus_id: str) -> list[Document]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Document).where(Document.corpus_id == corpus_id).order_by(Document.created_at.desc())
            )
            return list(result.scalars().all())

    async def fail_orphaned(self) -> int:
        """Startup reconciliation (Phase 6 follow-up) — a killed process
        never runs run_ingestion()'s except block, so a Document row can
        be left frozen at a non-terminal status forever, and the
        frontend's ingest/status polling loop then polls it forever too.
        Called once from main.py's lifespan on boot: any row still
        in-flight when THIS process starts couldn't have been left by a
        task this process is running (nothing survives a restart), so
        it's safe to assume every such row belongs to a previous,
        now-dead process and mark it failed. Returns the count fixed, for
        a startup log line."""
        NON_TERMINAL = ("pending", "fetching", "parsing", "embedding", "indexing")
        async with AsyncSessionLocal() as session:
            async with session.begin():
                result = await session.execute(select(Document).where(Document.status.in_(NON_TERMINAL)))
                orphans = result.scalars().all()
                for doc in orphans:
                    doc.status = "failed"
                    doc.error = "Orphaned by server restart — ingestion process was interrupted mid-run."
                    doc.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
                return len(orphans)


postgres_memory = PostgresMemory()
document_store = DocumentStore()
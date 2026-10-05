"""
Postgres persistence: PostgresMemory manages chat history and cascade deletion, DocumentStore tracks each uploaded file's ingestion lifecycle.
"""
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, delete
from services.api.app.config import settings
from services.api.app.memory.models import ChatHistory, Feedback, Document

engine = create_async_engine(settings.DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


class PostgresMemory:
    """Manages persisting and reading conversation history."""

    async def add_message(self, corpus_id: str, role: str, content: str) -> int:
        """Persists one chat message.

        Args:
            corpus_id: The session's corpus id.
            role: "user" | "assistant" | "system".
            content: The message text.

        Returns:
            The new row's id.
        """
        async with AsyncSessionLocal() as session:
            async with session.begin():
                msg = ChatHistory(corpus_id=corpus_id, role=role, content=content)
                session.add(msg)
                await session.flush()
                return msg.id

    async def get_history(self, corpus_id: str, limit: int = 10):
        """Fetches the most recent messages for a corpus, in chronological order.

        Args:
            corpus_id: The session's corpus id.
            limit: Max number of messages to return.

        Returns:
            A list of ChatHistory rows, oldest first.
        """
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(ChatHistory)
                .where(ChatHistory.corpus_id == corpus_id)
                .order_by(ChatHistory.created_at.desc())
                .limit(limit)
            )
            return result.scalars().all()[::-1]

    async def delete_by_corpus_id(self, corpus_id: str) -> None:
        """Deletes every chat history, feedback, and document row for a corpus.

        Args:
            corpus_id: The session's corpus id.
        """
        async with AsyncSessionLocal() as session:
            async with session.begin():
                await session.execute(delete(ChatHistory).where(ChatHistory.corpus_id == corpus_id))
                await session.execute(delete(Feedback).where(Feedback.corpus_id == corpus_id))
                await session.execute(delete(Document).where(Document.corpus_id == corpus_id))


class DocumentStore:
    """CRUD for the documents table."""

    async def create_pending(self, file_id: str, corpus_id: str, filename: str, s3_key: str) -> None:
        """Creates a new document row at upload time.

        Args:
            file_id: Unique id for this upload.
            corpus_id: The owning session's corpus id.
            filename: The original uploaded filename.
            s3_key: The object storage key the file will be written to.
        """
        async with AsyncSessionLocal() as session:
            async with session.begin():
                session.add(Document(
                    file_id=file_id, corpus_id=corpus_id, filename=filename,
                    s3_key=s3_key, status="pending",
                ))

    async def set_status(self, file_id: str, status: str, error: Optional[str] = None) -> None:
        """Updates a document's ingestion stage.

        Args:
            file_id: The document to update.
            status: The new stage string.
            error: Error detail — set alongside status="failed".
        """
        async with AsyncSessionLocal() as session:
            async with session.begin():
                result = await session.execute(select(Document).where(Document.file_id == file_id))
                doc = result.scalar_one_or_none()
                if doc is None:
                    return
                doc.status = status
                if error is not None:
                    doc.error = error
                if status in ("complete", "failed"):
                    doc.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)

    async def set_complete(self, file_id: str, chunk_count: int) -> None:
        """Marks a document as fully ingested.

        Args:
            file_id: The document to update.
            chunk_count: Number of chunks produced during ingestion.
        """
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
        """Fetches one document by its file_id.

        Args:
            file_id: The document to look up.

        Returns:
            The matching Document row, or None.
        """
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(Document).where(Document.file_id == file_id))
            return result.scalar_one_or_none()

    async def list_by_corpus_id(self, corpus_id: str) -> list[Document]:
        """Lists every document for a corpus, newest first.

        Args:
            corpus_id: The session's corpus id.

        Returns:
            A list of Document rows.
        """
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Document).where(Document.corpus_id == corpus_id).order_by(Document.created_at.desc())
            )
            return list(result.scalars().all())

    async def fail_orphaned(self) -> int:
        """Marks any document stuck at a non-terminal status as failed — run once at startup.

        Returns:
            Number of rows marked failed.
        """
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
"""
SQLAlchemy ORM models for the Postgres-backed tables: chat history, feedback, and uploaded document state.
"""
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from datetime import datetime, timezone

Base = declarative_base()


class ChatHistory(Base):
    """Stores the raw conversation log."""
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    corpus_id = Column(String(255), index=True, nullable=False)  # also the multi-tenancy scope
    role = Column(String(50), nullable=False)  # "user" | "assistant" | "system"
    content = Column(Text, nullable=False)
    metadata_ = Column(JSON, default=lambda: {}, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


class Feedback(Base):
    """Stores user feedback (thumbs up/down) on assistant responses."""
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    corpus_id = Column(String(255), nullable=False)
    message_id = Column(Integer, nullable=False)
    score = Column(Integer, nullable=False)  # 1 or -1
    category = Column(String(50), nullable=True)  # e.g. "inaccurate" | "not_relevant" | "incomplete" | "harmful"
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


class Document(Base):
    """One row per uploaded file — tracks its ingestion status."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(String(64), unique=True, index=True, nullable=False)
    corpus_id = Column(String(255), index=True, nullable=False)
    filename = Column(String(512), nullable=False)
    s3_key = Column(String(1024), nullable=False)
    status = Column(String(50), nullable=False, default="pending")  # pending|fetching|parsing|embedding|indexing|complete|failed
    error = Column(Text, nullable=True)
    chunk_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    completed_at = Column(DateTime, nullable=True)
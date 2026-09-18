from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, Boolean, UniqueConstraint
from datetime import datetime, timezone

Base = declarative_base()

class ChatHistory(Base):
    """
    SQLAlchemy Model for the 'chat_history' table.
    Stores the raw conversation log for auditing and context.
    """
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # corpus_id (= session_id) links messages to a conversation thread AND
    # scopes multi-tenancy — one column, not two, since a session has
    # exactly one corpus/conversation. Derived server-side from the
    # session cookie, never client-supplied.
    corpus_id = Column(String(255), index=True, nullable=False)
    
    # Role: 'user', 'assistant', or 'system'
    role = Column(String(50), nullable=False)
    
    # The actual message content
    content = Column(Text, nullable=False)
    
    # Metadata: Token usage, latency, model version used
    metadata_ = Column(JSON, default=lambda: {}, nullable=True)
    
    # Timestamp
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

class Feedback(Base):
    """
    SQLAlchemy Model for the 'feedback' table.
    Stores user feedback about assistant responses.
    """
    __tablename__ = "feedback"
    
    id         = Column(Integer, primary_key=True, autoincrement=True)
    corpus_id  = Column(String(255), nullable=False)
    message_id = Column(Integer, nullable=False)
    score      = Column(Integer, nullable=False)   # 1 or -1
    # Pre-defined category on thumbs-down (Phase 6 follow-up) — e.g.
    # "inaccurate" | "not_relevant" | "incomplete" | "harmful". Optional:
    # thumbs-up carries no category, and a thumbs-down can be submitted
    # without one if the user skips it. Pre-defined categories are
    # cheaper to act on later than parsing free-text comment alone.
    category   = Column(String(50), nullable=True)
    comment    = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


class Document(Base):
    """
    SQLAlchemy Model for the 'documents' table (Phase 6).
    One row per uploaded file — the persisted state ingest/status and
    corpus/documents read from. Didn't exist before Phase 6: uploads and
    ingestion previously had no queryable record at all, just log lines.
    """
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # file_id from upload.py's presigned-URL response — also embedded in
    # the S3 key (uploads/{corpus_id}/{file_id}.{ext}), so pipeline.py can
    # recover it from the key alone without any new data being threaded
    # through the MinIO webhook payload.
    file_id = Column(String(64), unique=True, index=True, nullable=False)
    corpus_id = Column(String(255), index=True, nullable=False)
    filename = Column(String(512), nullable=False)
    s3_key = Column(String(1024), nullable=False)

    # Fine-grained stage string, not a fixed enum — lets the frontend show
    # "parsing" / "embedding" / "indexing" while polling, not just a flat
    # "processing". Mirrors the stage names pipeline.py already logs
    # internally (stage=fetch/parse_chunk/embed/index_qdrant/complete).
    # Terminal values: "complete" | "failed".
    status = Column(String(50), nullable=False, default="pending")
    error = Column(Text, nullable=True)
    chunk_count = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    completed_at = Column(DateTime, nullable=True)

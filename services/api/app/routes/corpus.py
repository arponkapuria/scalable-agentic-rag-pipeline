"""GET /api/v1/corpus/documents (Phase 6) — lists every Document row for
the caller's own corpus_id, for the chat UI's document panel."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional

from services.api.app.memory.postgres import document_store
from services.api.app.session.dependency import get_corpus_id

router = APIRouter()


class DocumentSummary(BaseModel):
    file_id: str
    filename: str
    status: str
    chunk_count: Optional[int] = None
    created_at: str


@router.get("/documents", response_model=list[DocumentSummary])
async def list_documents(corpus_id: str = Depends(get_corpus_id)):
    docs = await document_store.list_by_corpus_id(corpus_id)
    return [
        DocumentSummary(
            file_id=d.file_id,
            filename=d.filename,
            status=d.status,
            chunk_count=d.chunk_count,
            created_at=d.created_at.isoformat(),
        )
        for d in docs
    ]

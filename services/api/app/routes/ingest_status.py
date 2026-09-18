"""
GET /api/v1/ingest/status/{file_id} (Phase 6) — reads the Document row
pipeline.py writes to as ingestion progresses. Didn't exist before Phase
6: uploads previously had no queryable state at all client-side, just
server logs.

corpus_id ownership is checked here (not just existence) — a session
shouldn't be able to poll another corpus's document status by guessing a
file_id, same boundary every other data-touching route enforces via
get_corpus_id.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from services.api.app.memory.postgres import document_store
from services.api.app.session.dependency import get_corpus_id

router = APIRouter()


class IngestStatusResponse(BaseModel):
    file_id: str
    filename: str
    status: str  # pending | fetching | parsing | embedding | indexing | complete | failed
    error: Optional[str] = None
    chunk_count: Optional[int] = None


@router.get("/status/{file_id}", response_model=IngestStatusResponse)
async def get_ingest_status(
    file_id: str,
    corpus_id: str = Depends(get_corpus_id),
):
    doc = await document_store.get_by_file_id(file_id)
    if doc is None or doc.corpus_id != corpus_id:
        # Same 404 for "doesn't exist" and "belongs to another corpus" —
        # doesn't leak whether a given file_id exists to a caller who
        # doesn't own it.
        raise HTTPException(status_code=404, detail="Document not found")

    return IngestStatusResponse(
        file_id=doc.file_id,
        filename=doc.filename,
        status=doc.status,
        error=doc.error,
        chunk_count=doc.chunk_count,
    )

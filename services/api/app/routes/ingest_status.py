"""
Ingestion status endpoint: reads the document row that the ingestion pipeline updates stage-by-stage, so the frontend can poll upload progress instead of relying on server logs.
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
    """Returns the current ingestion status for one uploaded file.

    Args:
        file_id: The document to look up.
        corpus_id: The caller's session — ownership is checked, not just existence.

    Returns:
        IngestStatusResponse with the document's current stage.

    Raises:
        HTTPException: 404 if the document doesn't exist or belongs to another corpus.
    """
    doc = await document_store.get_by_file_id(file_id)
    if doc is None or doc.corpus_id != corpus_id:
        raise HTTPException(status_code=404, detail="Document not found")

    return IngestStatusResponse(
        file_id=doc.file_id,
        filename=doc.filename,
        status=doc.status,
        error=doc.error,
        chunk_count=doc.chunk_count,
    )
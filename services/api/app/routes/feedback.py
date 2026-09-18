from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from services.api.app.memory.postgres import AsyncSessionLocal
from services.api.app.memory.models import Feedback
from services.api.app.session.dependency import get_corpus_id

router = APIRouter()

class FeedbackRequest(BaseModel):
    message_id: int  # ID of the assistant ChatHistory row (returned as
                      # ChatResponse.message_id — Phase 6 follow-up)
    score: int        # 1 (thumbs up) or -1 (thumbs down)
    category: str | None = None  # e.g. "inaccurate" | "not_relevant" | "incomplete" | "harmful" — thumbs-down only
    comment: str | None = None

@router.post("/")
async def submit_feedback(
    req: FeedbackRequest,
    corpus_id: str = Depends(get_corpus_id)
):
    """
    Submit user feedback for an AI response. Uses the Feedback ORM model
    (Phase 6) — previously a raw text() INSERT that had drifted out of
    sync with memory/models.py's Feedback class (same table, columns
    matched, but the model itself was unused dead weight).
    """
    try:
        async with AsyncSessionLocal() as session:
            async with session.begin():
                session.add(Feedback(
                    corpus_id=corpus_id,
                    message_id=req.message_id,
                    score=req.score,
                    category=req.category,
                    comment=req.comment,
                ))
        return {"status": "recorded"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

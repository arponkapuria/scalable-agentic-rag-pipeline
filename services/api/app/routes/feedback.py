"""
Feedback endpoint: records a thumbs up/down (with optional category and comment) against a specific assistant response.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from services.api.app.memory.postgres import AsyncSessionLocal
from services.api.app.memory.models import Feedback
from services.api.app.session.dependency import get_corpus_id

router = APIRouter()


class FeedbackRequest(BaseModel):
    message_id: int  # the assistant ChatHistory row id, returned as ChatResponse.message_id
    score: int  # 1 (thumbs up) or -1 (thumbs down)
    category: str | None = None  # e.g. "inaccurate" | "not_relevant" | "incomplete" | "harmful" — thumbs-down only
    comment: str | None = None


@router.post("/")
async def submit_feedback(
    req: FeedbackRequest,
    corpus_id: str = Depends(get_corpus_id)
):
    """Records feedback for an assistant response.

    Args:
        req: The feedback payload.
        corpus_id: The caller's session, injected from the session cookie.

    Returns:
        A fixed acknowledgement dict.

    Raises:
        HTTPException: 500 if the write fails.
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
import asyncio
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from services.api.app.config import settings
from services.api.app.session.dependency import get_corpus_id
from services.api.app.memory.postgres import document_store
from libs.utils.s3_client import get_s3_client
import uuid

router = APIRouter()

# boto3 is synchronous, but presigning is fast/CPU-bound — fine to call via
# asyncio.to_thread below rather than needing an async S3 SDK.
s3_client = get_s3_client()

# Schemas
class PresignedURLRequest(BaseModel):
    filename: str
    content_type: str # e.g., "application/pdf"

class PresignedURLResponse(BaseModel):
    upload_url: str
    file_id: str
    s3_key: str
    corpus_id: str  # frontend needs this verbatim to set the matching
                     # x-amz-meta-corpus_id header on its PUT — it can't
                     # read the httpOnly session cookie itself (Phase 2's
                     # cookie is deliberately httpOnly, JS can't see it).

# endpoints
@router.post("/generate-presigned-url", response_model=PresignedURLResponse)
async def generate_upload_url(
    req: PresignedURLRequest,
    corpus_id: str = Depends(get_corpus_id)
):
    """
    Generates a secure, temporary URL for the frontend to upload a file directly to S3.
    Use case: Handling 1GB+ PDF/Video files without blocking the API server.
    """
    # 1. Generate a unique file ID (UUID) to prevent overwrites
    file_id = str(uuid.uuid4())
    extension = req.filename.split('.')[-1] if '.' in req.filename else "bin"
    s3_key = f"uploads/{corpus_id}/{file_id}.{extension}"

    try:
        # 2. Generate the Presigned URL
        # The frontend uses this URL with a PUT request.
        url = await asyncio.to_thread(
            s3_client.generate_presigned_url,
            ClientMethod='put_object',
            Params={
                'Bucket': settings.S3_BUCKET_NAME,
                'Key': s3_key,
                'ContentType': req.content_type,
                'Metadata': {
                    'original_filename': req.filename,
                    'corpus_id': corpus_id
                },
            },
            ExpiresIn=3600 # URL valid for 1 hour
        )

        # Persisted row for ingest/status + corpus/documents (Phase 6) —
        # created here, at issuance, not on webhook receipt. The webhook
        # only fires once the client's PUT actually completes, so status
        # is "pending" (not yet uploaded/processed) until then.
        await document_store.create_pending(file_id, corpus_id, req.filename, s3_key)

        return PresignedURLResponse(
            upload_url=url,
            file_id=file_id,
            s3_key=s3_key,
            corpus_id=corpus_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
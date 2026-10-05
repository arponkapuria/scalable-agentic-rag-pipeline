"""
Upload endpoint: issues a presigned S3/MinIO URL so the frontend can upload a file directly to object storage without routing the file bytes through the API server.
"""
import asyncio
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from services.api.app.config import settings
from services.api.app.session.dependency import get_corpus_id
from services.api.app.memory.postgres import document_store
from libs.utils.s3_client import get_s3_client
import uuid

router = APIRouter()

# boto3 is synchronous, but presigning is fast/CPU-bound — fine to call via asyncio.to_thread
# below rather than needing an async S3 SDK.
s3_client = get_s3_client()


class PresignedURLRequest(BaseModel):
    filename: str
    content_type: str  # e.g., "application/pdf"


class PresignedURLResponse(BaseModel):
    upload_url: str
    file_id: str
    s3_key: str
    # The frontend needs this verbatim to set the matching x-amz-meta-corpus_id header on its
    # PUT — it can't read the httpOnly session cookie itself.
    corpus_id: str


@router.post("/generate-presigned-url", response_model=PresignedURLResponse)
async def generate_upload_url(
    req: PresignedURLRequest,
    corpus_id: str = Depends(get_corpus_id)
):
    """Generates a presigned upload URL and creates the matching pending document row.

    Args:
        req: The upload request, with the original filename and content type.
        corpus_id: The caller's session, injected from the session cookie.

    Returns:
        PresignedURLResponse with the upload URL and tracking ids.

    Raises:
        HTTPException: 500 if URL generation fails.
    """
    file_id = str(uuid.uuid4())
    extension = req.filename.split('.')[-1] if '.' in req.filename else "bin"
    s3_key = f"uploads/{corpus_id}/{file_id}.{extension}"

    try:
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
            ExpiresIn=3600  # URL valid for 1 hour
        )

        # Created here, at issuance, not on webhook receipt — status stays "pending" until
        # the client's PUT actually completes and the webhook fires.
        await document_store.create_pending(file_id, corpus_id, req.filename, s3_key)

        return PresignedURLResponse(
            upload_url=url,
            file_id=file_id,
            s3_key=s3_key,
            corpus_id=corpus_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
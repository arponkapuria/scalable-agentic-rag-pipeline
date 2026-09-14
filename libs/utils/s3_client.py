"""
Single boto3 S3 client factory used by both upload.py (presigned URLs) and
the Ray ingestion pipeline (reading uploaded files). S3_ENDPOINT_URL is None
for real AWS; set to MinIO's local URL for dev/demo — everything else
(bucket, key layout, boto3 calls) stays identical either way.

Explicit Config(signature_version="s3v4", addressing_style="path") is
required for MinIO: recent boto3/botocore versions can default to
virtual-hosted-style addressing for custom endpoints, which breaks SigV4
presigned-URL signing against MinIO (manifests as SignatureDoesNotMatch on
the PUT, not at generation time — the URL looks fine, the signature just
doesn't match what MinIO recomputes). Real AWS doesn't need this forced,
but it's harmless there too, so applied unconditionally rather than
branching on S3_ENDPOINT_URL.
"""
import boto3
from botocore.config import Config
from services.api.app.config import settings


def get_s3_client():
    return boto3.client(
        "s3",
        region_name=settings.AWS_REGION,
        endpoint_url=settings.S3_ENDPOINT_URL,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID if settings.S3_ENDPOINT_URL else None,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY if settings.S3_ENDPOINT_URL else None,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def delete_corpus_objects(corpus_id: str) -> int:
    """Cascade-delete hook for session expiry (session/cleanup.py) —
    removes every object under uploads/{corpus_id}/ (pipeline.py's
    _corpus_id_from_key derives corpus_id from exactly this prefix, so
    it's the correct and complete scope for one tenant's uploaded files).
    Sync (boto3) — callers running in an async context should wrap this
    in asyncio.to_thread, same as everywhere else boto3 is used in this
    codebase. Returns the number of objects deleted, for logging."""
    client = get_s3_client()
    prefix = f"uploads/{corpus_id}/"
    paginator = client.get_paginator("list_objects_v2")
    deleted = 0
    for page in paginator.paginate(Bucket=settings.S3_BUCKET_NAME, Prefix=prefix):
        keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if keys:
            client.delete_objects(Bucket=settings.S3_BUCKET_NAME, Delete={"Objects": keys})
            deleted += len(keys)
    return deleted

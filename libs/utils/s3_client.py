"""
Builds a boto3 S3 client (pointed at MinIO in development or real AWS S3 in production) and provides a helper to delete every object under a corpus's upload prefix.
"""
import boto3
from botocore.config import Config
from services.api.app.config import settings


def get_s3_client():
    """Builds a boto3 S3 client, configured for path-style addressing so it also works against MinIO.

    Returns:
        A configured boto3 S3 client.
    """
    return boto3.client(
        "s3",
        region_name=settings.AWS_REGION,
        endpoint_url=settings.S3_ENDPOINT_URL,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID if settings.S3_ENDPOINT_URL else None,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY if settings.S3_ENDPOINT_URL else None,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def delete_corpus_objects(corpus_id: str) -> int:
    """Deletes every object under uploads/{corpus_id}/ — used when a session's data is purged.

    Args:
        corpus_id: The corpus/session whose uploaded objects should be removed.

    Returns:
        Number of objects deleted.
    """
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
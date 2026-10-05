"""
Parallel, multipart bulk uploader for a local directory into the project's S3/MinIO bucket — a dev utility for seeding test data, not part of the live app.

Usage: python scripts/bulk_upload_s3.py <local_dir> [bucket_name]
"""
import os
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from boto3.s3.transfer import TransferConfig
from tqdm import tqdm

from libs.utils.s3_client import get_s3_client
from services.api.app.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def upload_directory(dir_path: str, bucket_name: str, max_workers: int = 10) -> None:
    """Recursively uploads a local directory to S3/MinIO, preserving folder structure.

    Args:
        dir_path: Local directory to upload.
        bucket_name: Target bucket.
        max_workers: Parallel upload threads.
    """
    s3 = get_s3_client()  # MinIO-aware — uses S3_ENDPOINT_URL/credentials from settings, not real AWS by default

    config = TransferConfig(
        multipart_threshold=25 * 1024 * 1024,
        multipart_chunksize=25 * 1024 * 1024,
        max_concurrency=20,
        use_threads=True,
    )

    files_to_upload = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            local_path = os.path.join(root, file)
            s3_path = os.path.relpath(local_path, dir_path)
            files_to_upload.append((local_path, s3_path))

    if not files_to_upload:
        logger.warning("No files found to upload.")
        return

    logger.info(f"Found {len(files_to_upload)} files. Starting upload...")

    def upload_file(local_path: str, s3_path: str, retries: int = 3) -> bool:
        for attempt in range(retries):
            try:
                s3.upload_file(Filename=local_path, Bucket=bucket_name, Key=s3_path, Config=config)
                return True
            except Exception as e:
                logger.warning(f"Retry {attempt + 1}/{retries} failed for {s3_path}: {e}")
        logger.error(f"Upload failed permanently: {s3_path}")
        return False

    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(upload_file, local, remote) for local, remote in files_to_upload]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Uploading"):
            if future.result():
                success_count += 1

    logger.info(f"Upload complete: {success_count}/{len(files_to_upload)} successful")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <local_dir> [bucket_name]")
        sys.exit(1)

    local_dir = sys.argv[1]
    bucket = sys.argv[2] if len(sys.argv) > 2 else settings.S3_BUCKET_NAME
    upload_directory(local_dir, bucket)
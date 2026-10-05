"""
Enriches a chunk's base metadata with a content hash and ingestion timestamp.
"""
import hashlib
import datetime


def enrich_metadata(base_metadata: dict, content: str) -> dict:
    """Adds a content hash, ingestion timestamp, and length to a chunk's metadata.

    Args:
        base_metadata: The chunk's existing metadata dict.
        content: The chunk's text content.

    Returns:
        A new dict with "chunk_hash", "ingested_at", and "length" added.
    """
    content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
    ingestion_time = datetime.datetime.utcnow().isoformat()

    enriched = base_metadata.copy()
    enriched.update({
        "chunk_hash": content_hash,
        "ingested_at": ingestion_time,
        "length": len(content)
    })
    return enriched
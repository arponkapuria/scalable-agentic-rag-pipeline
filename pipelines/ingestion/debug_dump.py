"""
Writes intermediate ingestion output (parsed chunks, captioned figures) to disk, so output quality can be inspected directly instead of inferred from log line counts. Gated by INGEST_DEBUG_DUMP — turn off before a public deploy, since this writes raw document content to local disk.
"""
import json
import logging
import os
from typing import Any, Optional

from services.api.app.config import settings

logger = logging.getLogger(__name__)


def dump_debug_artifact(corpus_id: str, stage: str, data: Any) -> Optional[str]:
    """Writes one debug artifact as JSON, if debug dumping is enabled.

    Args:
        corpus_id: Used to scope the output path.
        stage: Name for this artifact (e.g. "chunks", "docling_pictures").
        data: JSON-serializable data to write.

    Returns:
        The path written, or None if dumping is disabled or the write failed.
    """
    if not settings.INGEST_DEBUG_DUMP:
        return None
    try:
        out_dir = os.path.join(settings.INGEST_DEBUG_DIR, corpus_id)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{stage}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        logger.info(f"[ingest:{corpus_id}] debug artifact written: {path}")
        return path
    except Exception as e:
        logger.warning(f"[ingest:{corpus_id}] failed to write debug artifact ({stage}): {e}")
        return None
"""
Optional debug artifacts for ingestion — writes intermediate pipeline
output (parsed chunks, extracted graph data) to disk so output QUALITY
can be inspected directly, not just inferred from log line counts.
Gated by INGEST_DEBUG_DUMP (default on for local dev; turn off before the
public demo — this writes raw document content to local disk).
"""
import json
import logging
import os
from typing import Any, Optional

from services.api.app.config import settings

logger = logging.getLogger(__name__)


def dump_debug_artifact(corpus_id: str, stage: str, data: Any) -> Optional[str]:
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
        # Never let a debug convenience break real ingestion.
        logger.warning(f"[ingest:{corpus_id}] failed to write debug artifact ({stage}): {e}")
        return None

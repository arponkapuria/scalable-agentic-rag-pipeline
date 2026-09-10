import logging
import logging.handlers
import json
import os
import sys
from datetime import datetime

from services.api.app.config import settings

class JSONFormatter(logging.Formatter):
    """
    Formats log records as a JSON object.
    Includes timestamp, level, and message.
    """
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
            
        # Add extra fields (e.g., user_id, trace_id) passed via 'extra' dict
        if hasattr(record, "request_id"):
            log_record["request_id"] = record.request_id
            
        return json.dumps(log_record)

class ConsoleFormatter(logging.Formatter):
    """level + message only — readable in a dev terminal. Console stays
    human-readable; the file handler below is where the structured,
    parseable copy lives."""
    def format(self, record):
        line = f"{record.levelname}: {record.getMessage()}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line

def setup_logging():
    """
    Console: plain level+message, human-readable, real TTY (so uvicorn's
    own colored access-log output — method/path/status — keeps working
    natively; don't pipe `make dev` through `tee`/`grep` to get logs, use
    the file below instead).

    File: rotating, JSON-structured, one line per record — the standard
    split (humans read the console, tools/log-aggregation read the file).
    Rotates by size so a long-running dev/demo process doesn't grow an
    unbounded log file; old files are numbered suffixes
    (api.log.1, api.log.2, ...), oldest deleted past LOG_BACKUP_COUNT.
    """
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ConsoleFormatter())

    os.makedirs(settings.LOG_DIR, exist_ok=True)
    log_file_path = os.path.join(settings.LOG_DIR, settings.LOG_FILE_NAME)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path,
        maxBytes=settings.LOG_MAX_BYTES,
        backupCount=settings.LOG_BACKUP_COUNT,
    )
    file_handler.setFormatter(JSONFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove default handlers to avoid duplicate logs as frameworks like FastAPI/uvicorn pre-attach their own handlers
    if root_logger.handlers:
        root_logger.handlers = []
        
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Silence noisy libraries. uvicorn.access is left ENABLED — it's the
    # only thing that prints the HTTP method/path/status line per
    # request, and it attaches its own handler/formatter directly to this
    # logger (independent of root), which is what gives it native color
    # in a real terminal.
    logging.getLogger("httpx").setLevel(logging.WARNING)

# Initialize on import
setup_logging()
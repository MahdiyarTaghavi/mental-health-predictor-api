import logging
import logging.config
import inspect
import os
import json
from datetime import datetime
from glob import glob
from typing import Any

from src.utils.loggings.logging_context import request_context

# -------- Daily Rotating Handler --------
class DailyRotatingFileHandler(logging.FileHandler):
    """
    Creates one log file per day: YYYY-MM-DD_<service>.log
    Keeps only `backupCount` most recent files.
    """
    def __init__(self, base_dir: str, service_name: str, backupCount: int = 100, encoding=None):
        self.base_dir = base_dir
        self.service_name = service_name
        self.backupCount = backupCount
        os.makedirs(base_dir, exist_ok=True)
        self.current_date = datetime.utcnow().strftime("%Y-%m-%d")
        filename = self._make_filename(self.current_date)
        super().__init__(filename, encoding=encoding)

    def _make_filename(self, date_str: str) -> str:
        return os.path.join(self.base_dir, f"{date_str}_{self.service_name}.log")

    def emit(self, record):
        date_now = datetime.utcnow().strftime("%Y-%m-%d")
        if date_now != self.current_date:
            # date changed -> switch file
            self.current_date = date_now
            self.close()
            self.baseFilename = self._make_filename(date_now)
            self.stream = self._open()
            self._cleanup_old_files()
        super().emit(record)

    def _cleanup_old_files(self):
        pattern = os.path.join(self.base_dir, f"*_{self.service_name}.log")
        files = sorted(glob(pattern))
        if len(files) > self.backupCount:
            for old in files[0:len(files)-self.backupCount]:
                try:
                    os.remove(old)
                except OSError:
                    pass

class JSONFormatter(logging.Formatter):
    """
    Format log record as a JSON string
    """
    def format(self, record):
        # Start with standard attributes
        log_record = {
            "level": record.levelname,
            "message": record.getMessage(),
            "timestamp": datetime.utcnow().isoformat() + "Z",  # ISO 8601
        }

        # Include our extra "custom" dict if present
        if hasattr(record, "custom"):
            log_record.update(record.custom)

        return json.dumps(log_record)


# -------- Structured Logger --------
class StructuredLogger:
    """
    Wrapper to add metadata: timestamp, file, function, plus user-provided kwargs
    """
    def __init__(self, service_name: str):
        self.logger = logging.getLogger(service_name)
        self.service_name = service_name

    def _extra(self, **kwargs) -> dict:
        frame = inspect.stack()[2]  # caller
        req_info = request_context.get() or {}

        return {
            "service": self.service_name,
            "file": os.path.basename(frame.filename),
            "function": frame.function,
            "endpoint": req_info.get("endpoint"),
            "method": req_info.get("method"),
            "client_ip": req_info.get("client_ip"),
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }

    def debug(self, msg: str, **kwargs: Any):
        self.logger.debug(msg, extra={"custom": self._extra(**kwargs)})

    def info(self, msg: str, **kwargs: Any):
        self.logger.info(msg, extra={"custom": self._extra(**kwargs)})

    def warning(self, msg: str, **kwargs: Any):
        self.logger.warning(msg, extra={"custom": self._extra(**kwargs)})

    def error(self, msg: str, **kwargs: Any):
        self.logger.error(msg, extra={"custom": self._extra(**kwargs)})

    def critical(self, msg: str, **kwargs: Any):
        self.logger.critical(msg, extra={"custom": self._extra(**kwargs)})

    def exception(self, msg: str, **kwargs: Any):
        self.logger.exception(msg, extra={"custom": self._extra(**kwargs)})
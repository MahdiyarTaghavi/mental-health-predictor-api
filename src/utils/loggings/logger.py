import logging.config
from pathlib import Path

from src.utils.loggings.logging_handlers import StructuredLogger

BASE_DIR = Path(__file__).resolve().parents[3]   # src/utils → src → project root

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "src.utils.loggings.logging_handlers.JSONFormatter"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "json",
        },
        "app": {
            "class": "src.utils.loggings.logging_handlers.DailyRotatingFileHandler",
            "level": "DEBUG",
            "formatter": "json",
            "base_dir": f"{BASE_DIR}/logs",
            # "base_dir": "/var/logs",
            "service_name": "app",
            "backupCount": 100,
        },
    },
    "loggers": {
        "app": {"handlers": ["app", "console"], "level": "DEBUG", "propagate": False},
    },
}

logging.config.dictConfig(LOGGING_CONFIG)

app_log = StructuredLogger("app")
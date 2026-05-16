import logging.config

from utils.loggings.logging_handlers import StructuredLogger


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "utils.loggings.logging_handlers.JSONFormatter"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "json",
        },
        "app": {
            "class": "utils.loggings.logging_handlers.DailyRotatingFileHandler",
            "level": "DEBUG",
            "formatter": "json",
            "base_dir": "/var/logs/mental-health-predictor",
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
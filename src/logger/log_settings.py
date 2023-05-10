from pathlib import Path

path = Path(__file__).resolve().parents[2]

LOG_FILENAME = path / "logs/log.log"

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s : %(filename)s[LINE:%(lineno)d]"
            " : %(levelname)s : %(message)s",
        },
    },
    "handlers": {
        "logfile": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "filename": LOG_FILENAME,
            "formatter": "default",
        },
        "verbose_output": {
            "formatter": "default",
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "info_logger": {
            "level": "DEBUG",
            "handlers": ["verbose_output"],
        },
        "file_logger": {
            "level": "DEBUG",
            "handlers": ["logfile", "verbose_output"],
        },
    },
}

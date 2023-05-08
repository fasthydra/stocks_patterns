from pathlib import Path

path = Path(__file__).resolve().parents[2]

DATA_DIVISION_LOG_FILENAME = path / "logs/data_division.log"

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
        "logfile_data_div": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "filename": DATA_DIVISION_LOG_FILENAME,
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
            "level": "INFO",
            "handlers": ["verbose_output"],
        },
        "data_division": {
            "level": "INFO",
            "handlers": ["logfile_data_div", "verbose_output"],
        },
    },
}

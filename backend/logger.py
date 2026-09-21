"""
logger.py
---------
One shared logger for the whole project.

Logs go to the console AND to data/cognify.log so that you can show the full
pipeline trace during a demo or viva.  Only pipeline information is logged -
there are no API keys or secrets anywhere in this project.
"""

import logging
import sys

from backend import config

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "cognify") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:            # already configured
        return logger

    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
    formatter = logging.Formatter(_LOG_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    try:
        file_handler = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # Never crash because a log file cannot be created
        logger.warning("Could not open log file %s - logging to console only", config.LOG_FILE)

    logger.propagate = False
    return logger


def shorten(text: str, limit: int = 240) -> str:
    """Helper for readable log lines."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + " ..."

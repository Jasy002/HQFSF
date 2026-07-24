"""
Central logging utility for HQFSF.
"""

import logging
from pathlib import Path

LOG_DIRECTORY = Path("logs")
LOG_DIRECTORY.mkdir(exist_ok=True)

LOG_FILE = LOG_DIRECTORY / "hqfsf.log"


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a configured logger.
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_FILE)

    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.addHandler(console_handler)

    return logger
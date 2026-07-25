"""
Logging utility for HQFSF.
"""

import logging
import os


def setup_logger(
    log_file: str = "logs/hqfsf.log",
    level: int = logging.INFO
) -> logging.Logger:

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger("HQFSF")

    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_file)

    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.addHandler(console_handler)

    return logger
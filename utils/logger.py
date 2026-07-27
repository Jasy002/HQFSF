"""
logger.py
=========

Centralized logging utility for the
Hybrid Quantum Feature Selection Framework (HQFSF).
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ==========================================================
# LOG DIRECTORY
# ==========================================================

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "hqfsf.log"

# ==========================================================
# LOG FORMAT
# ==========================================================

LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)-8s | "
    "%(name)s | "
    "%(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

FORMATTER = logging.Formatter(
    fmt=LOG_FORMAT,
    datefmt=DATE_FORMAT,
)

# ==========================================================
# LOGGER FACTORY
# ==========================================================

def get_logger(
    name: str,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create and return a configured logger.
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    # ------------------------------------------
    # Console Handler
    # ------------------------------------------

    console_handler = logging.StreamHandler()

    console_handler.setLevel(level)

    console_handler.setFormatter(FORMATTER)

    # ------------------------------------------
    # Rotating File Handler
    # ------------------------------------------

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )

    file_handler.setLevel(level)

    file_handler.setFormatter(FORMATTER)

    # ------------------------------------------
    # Add Handlers
    # ------------------------------------------

    logger.addHandler(console_handler)

    logger.addHandler(file_handler)

    logger.propagate = False

    return logger


# ==========================================================
# PROJECT LOGGER
# ==========================================================

project_logger = get_logger("HQFSF")


# ==========================================================
# CLEAR LOG FILE
# ==========================================================

def clear_logs():
    """
    Clear the current log file.
    """

    with open(
        LOG_FILE,
        "w",
        encoding="utf-8",
    ):
        pass


# ==========================================================
# LOG BANNER
# ==========================================================

def log_banner(
    title: str,
    logger: logging.Logger | None = None,
):
    """
    Print a banner in the log.
    """

    if logger is None:

        logger = project_logger

    line = "=" * 80

    logger.info(line)

    logger.info(title)

    logger.info(line)


# ==========================================================
# LOG SEPARATOR
# ==========================================================

def log_separator(
    logger: logging.Logger | None = None,
):
    """
    Print a separator line.
    """

    if logger is None:

        logger = project_logger

    logger.info("-" * 80)


# ==========================================================
# SET LOG LEVEL
# ==========================================================

def set_log_level(
    level: int,
    logger: logging.Logger | None = None,
):
    """
    Update logger level.
    """

    if logger is None:

        logger = project_logger

    logger.setLevel(level)

    for handler in logger.handlers:

        handler.setLevel(level)


# ==========================================================
# REPRESENTATION
# ==========================================================

def __repr__():

    return (
        f"HQFSF Logger "
        f"(file='{LOG_FILE}')"
    )
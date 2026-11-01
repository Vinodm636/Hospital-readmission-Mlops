"""
logger.py
---------
Centralised logging configuration for the Hospital Readmission project.

Why we need this:
    A shared logger ensures every module (ingestion, cleaning, validation,
    loading, API) writes structured log messages to the same log files and
    console format.  This makes debugging much easier.

How to use:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Processing started")
    logger.error("Something went wrong: %s", str(e))
"""

import logging
import os
from datetime import datetime
from pathlib import Path


def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Create and return a named logger that writes to both console and a
    date-stamped log file.

    Args:
        name (str): Logger name — typically pass __name__ from the caller.
        log_dir (str): Directory where log files are stored.
                       Defaults to ./logs relative to project root.

    Returns:
        logging.Logger: Configured logger instance.

    Example:
        logger = get_logger("ingestion")
        logger.info("File loaded: %s rows", 101766)
    """
    # Ensure the log directory exists
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Build a dated log file name
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = log_path / f"pipeline_{today}.log"

    # Create the logger
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if this function is called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # ----------------------------------------------------------------
    # Format: timestamp | level | module name | message
    # ----------------------------------------------------------------
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ----------------------------------------------------------------
    # Console handler — shows INFO and above
    # ----------------------------------------------------------------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # ----------------------------------------------------------------
    # File handler — captures DEBUG and above for full audit trail
    # ----------------------------------------------------------------
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def get_pipeline_logger(pipeline_name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Convenience wrapper that creates a logger with a pipeline-specific
    log file alongside the shared daily log.

    Args:
        pipeline_name (str): Logical name for the pipeline run.
        log_dir (str): Directory for log files.

    Returns:
        logging.Logger: Configured logger.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    log_file = log_path / f"{pipeline_name}_{today}.log"

    logger = logging.getLogger(pipeline_name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

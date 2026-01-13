"""
training_related.py – Optimized logging utility for the training pipeline.

Replaces the naive read‑modify‑write approach with Python's built‑in
`logging` module plus a rotating file handler for resilience and speed.

Usage example
-------------
>>> from training_related import setup_logger
>>> logger = setup_logger('./working_dir')
>>> logger.info('Training started')

The legacy ``save_to_log`` signature is still supported so you can drop this
file in without touching the rest of the codebase.
"""

from __future__ import annotations

import logging
import os
import sys

project_root = '../'
sys.path.append(project_root)
from logging.handlers import RotatingFileHandler
from pathlib import Path

__all__ = [
    "setup_logger",
    "save_to_log",
]


def setup_logger(
    working_dir: str | Path,
    filename: str = "log.txt",
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB per segment
    backup_count: int = 5,
    console: bool = True,
) -> logging.Logger:
    """Return a configured :class:`logging.Logger` instance.

    Parameters
    ----------
    working_dir : str | Path
        Directory where the log file lives. Created if it does not exist.
    filename : str, optional
        Name of the log file, by default ``"log.txt"``.
    level : int, optional
        Logging level, by default :pydata:`logging.INFO`.
    max_bytes : int, optional
        Rotate the log when it grows past this size (in bytes), by default
        10 MB.
    backup_count : int, optional
        How many rotated log files to keep.
    console : bool, optional
        If *True*, also stream logs to *stdout*.
    """
    workdir = Path(working_dir)

    logger_name = f"training_logger_{workdir.resolve()}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Idempotent: if this logger already has handlers, just return it.
    if logger.handlers:
        return logger

    log_path = workdir / filename
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # Prevent duplicate logs if the root logger also has handlers.
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Legacy shim
# ---------------------------------------------------------------------------

def save_to_log(msg: str, working_dir: str | Path, print_msg: bool = False) -> None:
    """Backward‑compatible wrapper around :pyfunc:`setup_logger`.

    Keeps the original signature so existing calls do not need to change.
    Internally, we delegate to a properly configured logger.
    """
    logger = setup_logger(working_dir, console=print_msg)
    logger.info(msg)
"""
Logging utilities for the Flowers102 classification project.
"""

import logging
import os
import sys
from pathlib import Path


def setup_logging(log_file=None, log_level=logging.INFO):
    """
    Set up logging configuration.

    Args:
        log_file: Path to log file, if None, logs only to console
        log_level: Logging level
    """
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

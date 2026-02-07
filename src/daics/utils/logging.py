from __future__ import annotations

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Minimal, consistent logger for the whole project.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger

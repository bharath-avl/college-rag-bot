"""
Logging utility configuration.

Provides a pre-configured logger with console output and a consistent
format across the entire backend.
"""

import logging
import sys

_LOG_FORMAT: str = "%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s"
_LOG_LEVEL: int = logging.INFO
_CONFIGURED: bool = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.

    The root handler is set up once on first call so that every logger
    shares the same console formatter.

    Args:
        name: Dotted module name (e.g. ``"backend.rag.vectorstore"``).

    Returns:
        A ``logging.Logger`` ready for use.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logging.root.addHandler(handler)
        logging.root.setLevel(_LOG_LEVEL)
        _CONFIGURED = True

    return logging.getLogger(name)

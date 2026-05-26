"""
Logging utility configuration.
"""

import logging

def get_logger(name: str) -> logging.Logger:
    """
    Get configured logger instance.
    """
    logger = logging.getLogger(name)
    return logger

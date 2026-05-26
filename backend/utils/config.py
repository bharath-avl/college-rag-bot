"""
Configuration and environment variable loader.
"""

import os
from dotenv import load_dotenv

load_dotenv()

def get_env_var(key: str, default: str | None = None) -> str:
    """
    Get environment variable or default value.
    Raises ValueError if key is not found and no default is provided.
    """
    val = os.getenv(key, default)
    if val is None:
        raise ValueError(f"Environment variable {key} not found")
    return val

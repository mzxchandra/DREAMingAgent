"""Globus authentication wrapper for ALCF access."""

import sys
from pathlib import Path

# Add parent directory to path to import inference_auth_token
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference_auth_token import get_access_token as _get_access_token


def get_alcf_token() -> str:
    """
    Get a fresh ALCF access token using Globus authentication.

    This function wraps the inference_auth_token module to provide
    a clean interface for the Research Agent.

    Returns:
        str: Valid ALCF access token

    Raises:
        Exception: If authentication fails or token is invalid
    """
    try:
        token = _get_access_token()
        if not token:
            raise ValueError("Empty token returned from authentication")
        return token
    except Exception as e:
        raise Exception(
            f"Failed to get ALCF token: {e}. "
            "Please run 'python inference_auth_token.py authenticate' to set up authentication."
        ) from e


def check_token_validity() -> bool:
    """
    Check if the current token is valid without throwing an exception.

    Returns:
        bool: True if token is valid, False otherwise
    """
    try:
        token = get_alcf_token()
        return len(token) > 0
    except Exception:
        return False

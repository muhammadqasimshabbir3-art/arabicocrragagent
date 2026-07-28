"""Compatibility shim — prefer `from utils import run_in_thread`."""

from utils.async_utils import run_in_thread

__all__ = ["run_in_thread"]

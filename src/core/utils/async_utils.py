"""Helpers for running blocking code without blocking the LangGraph event loop."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


async def run_in_thread(func: Callable[..., T], /, *args, **kwargs) -> T:
    """Run a blocking function in a worker thread."""
    return await asyncio.to_thread(func, *args, **kwargs)

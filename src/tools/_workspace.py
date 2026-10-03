"""Internal workspace path helpers for file/git/terminal tools."""
from __future__ import annotations

import os
from pathlib import Path


def _resolve_workspace_path(filepath: str) -> Path:
    """Resolve a filepath relative to PROJECT_DIR env var or the repo root."""
    base = Path(os.getenv("PROJECT_DIR", Path(__file__).resolve().parents[2]))
    p = Path(filepath)
    return p if p.is_absolute() else base / p


def _truncate_text(text: str, max_chars: int = 8000) -> str:
    """Truncate text to max_chars with a note about how much was cut."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... (truncated {len(text) - max_chars} chars)"

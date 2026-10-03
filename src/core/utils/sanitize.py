"""Filename and collection-name sanitization for safe indexing/citations."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

_COLLECTION_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
_UNSAFE_NAME_CHARS = re.compile(r"[\x00-\x1f\x7f<>:\"|?*\\/]+")


def sanitize_filename(name: str | None, *, default: str = "uploaded.bin") -> str:
    """Return a basename-only safe display filename (no path traversal)."""
    raw = (name or "").strip() or default
    # Normalize Unicode, strip path components, reject null/control chars.
    raw = unicodedata.normalize("NFKC", raw)
    base = Path(raw.replace("\\", "/")).name
    base = _UNSAFE_NAME_CHARS.sub("_", base).strip(" .")
    if not base or base in {".", ".."}:
        return default
    # Cap length while keeping extension when possible.
    if len(base) > 200:
        suffix = Path(base).suffix[:20]
        stem = Path(base).stem[: 200 - len(suffix)]
        base = f"{stem}{suffix}"
    return base


def validate_collection_name(name: str) -> str:
    """Allow only safe vector-store collection identifiers."""
    cleaned = (name or "").strip()
    if not _COLLECTION_RE.match(cleaned):
        raise ValueError(
            "Invalid collection name. Use 1–64 chars: letters, digits, "
            "underscore, hyphen (must start with alphanumeric)."
        )
    return cleaned

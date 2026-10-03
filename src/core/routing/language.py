"""Lightweight language detection for Arabic / English / mixed."""

from __future__ import annotations

import re

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    """Return 'ar', 'en', or 'mixed' based on script presence."""
    if not text or not text.strip():
        return "en"
    arabic = len(_ARABIC_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    if arabic and latin:
        # Treat as mixed when both scripts are meaningful.
        if arabic >= 3 and latin >= 3:
            return "mixed"
        return "ar" if arabic >= latin else "en"
    if arabic:
        return "ar"
    return "en"


def has_arabic(text: str) -> bool:
    """Return True if the text contains Arabic letters."""
    return bool(_ARABIC_RE.search(text or ""))

"""Intent helpers retained for compatibility with older imports."""

from routing.intent import get_latest_user_text, pick_route, wants_summary
from routing.language import detect_language, has_arabic

__all__ = [
    "detect_language",
    "get_latest_user_text",
    "has_arabic",
    "pick_route",
    "wants_summary",
]

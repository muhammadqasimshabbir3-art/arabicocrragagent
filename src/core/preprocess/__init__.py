"""OCR preprocessing package."""

from .arabic_normalize import clean_ocr_text, dedupe_repeated_lines, normalize_arabic

__all__ = ["clean_ocr_text", "dedupe_repeated_lines", "normalize_arabic"]

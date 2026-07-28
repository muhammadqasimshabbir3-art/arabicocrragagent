"""OCR engine abstractions and backends."""

from .base import OCRDocument, OCREngine, OCRPage, OCRError
from .factory import get_ocr_engine

__all__ = ["OCRDocument", "OCREngine", "OCRPage", "OCRError", "get_ocr_engine"]

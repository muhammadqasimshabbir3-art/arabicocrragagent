"""Text extraction subagent — digital PDF / TXT / DOCX only."""

from subagents.ocr.base import OCRDocument, OCREngine, OCRError, OCRPage
from subagents.ocr.factory import extract_document_text, get_ocr_engine

__all__ = [
    "get_ocr_engine",
    "extract_document_text",
    "OCRDocument",
    "OCREngine",
    "OCRError",
    "OCRPage",
]

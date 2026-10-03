"""Text-extraction factory — digital text only (PDF text layer / TXT / DOCX)."""

from __future__ import annotations

from functools import lru_cache

from core.loaders.base import LoadedDocument
from core.utils.logging_utils import get_logger
from subagents.ocr.base import OCRDocument, OCREngine, OCRError
from subagents.ocr.digital_text import DigitalTextEngine

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_ocr_engine(name: str | None = None) -> OCREngine:
    """Return the digital text engine (only supported backend)."""
    engine_name = (name or "digital").lower()
    if engine_name not in {"digital", "digital_text", "none", ""}:
        raise OCRError(
            f"Unknown text engine: {engine_name}. "
            "This agent supports digital extraction only."
        )
    return DigitalTextEngine()


def extract_document_text(document: LoadedDocument) -> OCRDocument:
    """Extract text from a text-layer PDF, TXT, or DOCX document."""
    digital = DigitalTextEngine()
    if document.needs_ocr:
        raise OCRError(
            "This document appears scanned or image-based. "
            "Upload a text-layer PDF, TXT, or DOCX."
        )
    try:
        return digital.extract(document)
    except OCRError:
        logger.info("Digital text extraction failed for %s", document.filename)
        raise

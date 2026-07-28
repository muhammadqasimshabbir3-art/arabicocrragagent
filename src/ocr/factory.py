"""OCR engine factory — select backend via configuration."""

from __future__ import annotations

from functools import lru_cache

from config.settings import get_settings
from loaders.base import LoadedDocument
from ocr.base import OCRDocument, OCREngine, OCRError
from ocr.digital_text import DigitalTextEngine
from preprocess.arabic_normalize import looks_unreliable_arabic_extraction
from utils.logging_utils import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=4)
def get_ocr_engine(name: str | None = None) -> OCREngine:
    """Return an OCR engine instance (cached by name)."""
    settings = get_settings()
    engine_name = (name or settings.ocr_engine or "qari").lower()

    if engine_name in {"digital", "digital_text", "none"}:
        return DigitalTextEngine()

    if engine_name == "qari":
        from ocr.qari import QariOCREngine

        return QariOCREngine()

    if engine_name == "ain":
        # Same VLM interface path; override OCR_MODEL_ID=MBZUAI/AIN in env.
        from ocr.qari import QariOCREngine

        return QariOCREngine()

    raise OCRError(f"Unknown OCR engine: {engine_name}")


def extract_document_text(document: LoadedDocument) -> OCRDocument:
    """Prefer digital text; fall back to configured OCR for scans/images."""
    digital = DigitalTextEngine()
    if not document.needs_ocr:
        try:
            digital_doc = digital.extract(document)
            # Some PDFs contain selectable text with broken RTL order/encoding.
            # If extraction still looks unreliable, retry with OCR engine.
            joined = "\n".join(page.text for page in digital_doc.pages if page.text)
            if (
                document.mime_type == "application/pdf"
                and looks_unreliable_arabic_extraction(joined)
            ):
                logger.info("Digital PDF text quality looks unreliable; trying OCR fallback")
            else:
                return digital_doc
        except OCRError:
            logger.info("Digital text insufficient; falling back to OCR")

    engine = get_ocr_engine()
    if isinstance(engine, DigitalTextEngine):
        raise OCRError(
            "This document appears scanned/image-based, but OCR_ENGINE=digital. "
            "Set OCR_ENGINE=qari to enable Arabic OCR."
        )
    return engine.extract(document)

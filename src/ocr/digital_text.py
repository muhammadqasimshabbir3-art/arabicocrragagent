"""Digital text path — use embedded PDF text when available (no OCR model)."""

from __future__ import annotations

from loaders.base import LoadedDocument
from ocr.base import OCRDocument, OCREngine, OCRError, OCRPage
from preprocess.arabic_normalize import clean_ocr_text, dedupe_repeated_lines


class DigitalTextEngine(OCREngine):
    """Extract and normalize digital text layers without running a VLM."""

    name = "digital_text"

    def extract(self, document: LoadedDocument) -> OCRDocument:
        pages: list[OCRPage] = []
        for page in document.pages:
            text = dedupe_repeated_lines(clean_ocr_text(page.text))
            pages.append(
                OCRPage(
                    page_number=page.page_number,
                    text=text,
                    html="",
                    metadata={"source": "digital_text"},
                )
            )
        if not any(page.text.strip() for page in pages):
            raise OCRError(
                "No extractable digital text found. A scanned document requires OCR."
            )
        return OCRDocument(
            filename=document.filename,
            fingerprint=document.fingerprint,
            pages=pages,
            engine=self.name,
            metadata={"page_count": len(pages)},
        )

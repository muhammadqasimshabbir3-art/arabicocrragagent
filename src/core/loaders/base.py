"""Shared loader data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class LoaderError(ValueError):
    """Raised when a document cannot be loaded safely."""


@dataclass(frozen=True)
class LoadedPage:
    """One page or image frame ready for OCR or text extraction."""

    page_number: int
    text: str = ""
    image_bytes: bytes | None = None
    width: int | None = None
    height: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_text(self) -> bool:
        return bool(self.text and self.text.strip())


@dataclass
class LoadedDocument:
    """Normalized document prior to OCR / indexing."""

    filename: str
    mime_type: str
    fingerprint: str
    pages: list[LoadedPage]
    raw_bytes: bytes
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_ocr(self) -> bool:
        """True when pages lack extractable digital text."""
        if not self.pages:
            return True
        fmt = str(self.metadata.get("format") or "").lower()
        # Plain text / DOCX always use the digital path when any text exists.
        if fmt in {"text", "docx"} or self.mime_type.startswith("text/"):
            return not any(page.has_text for page in self.pages)
        text_chars = sum(len(page.text.strip()) for page in self.pages)
        return text_chars < 40

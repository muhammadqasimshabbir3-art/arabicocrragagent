"""OCR data models and abstract engine interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from loaders.base import LoadedDocument


class OCRError(RuntimeError):
    """Raised when OCR extraction fails."""


@dataclass
class OCRPage:
    """OCR result for a single page."""

    page_number: int
    text: str
    html: str = ""
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OCRDocument:
    """Full-document OCR output optimized for RAG."""

    filename: str
    fingerprint: str
    pages: list[OCRPage]
    engine: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        parts: list[str] = []
        for page in self.pages:
            body = (page.text or "").strip()
            if not body:
                continue
            parts.append(f"[Page {page.page_number}]\n{body}")
        return "\n\n".join(parts).strip()


class OCREngine(ABC):
    """Abstract OCR backend. Downstream code depends only on this interface."""

    name: str = "base"

    @abstractmethod
    def extract(self, document: LoadedDocument) -> OCRDocument:
        """Extract text (and optional layout HTML) from a loaded document."""

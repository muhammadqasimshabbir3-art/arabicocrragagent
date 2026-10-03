"""PDF generation tool using PyMuPDF — supports Arabic (RTL) and English text."""
from __future__ import annotations

import html
import os
from pathlib import Path

try:
    from langchain.tools import tool
except ImportError:  # graceful fallback if langchain not installed
    def tool(fn):  # type: ignore[misc]
        return fn


def _split_pages(text: str, chars_per_page: int = 2800) -> list[str]:
    """Split long text into page-sized chunks."""
    text = text.strip()
    if not text:
        return [""]
    chunks: list[str] = []
    for start in range(0, len(text), chars_per_page):
        chunk = text[start : start + chars_per_page].strip()
        if chunk:
            chunks.append(chunk)
    return chunks or [""]


@tool
def create_pdf(text: str, output_path: str = "output.pdf", title: str = "") -> str:
    """Create a PDF from the given text.

    Supports Arabic (RTL) and English. Uses PyMuPDF which is already installed.
    Returns the absolute path to the saved PDF.

    Args:
        text: The text content for the PDF.
        output_path: Where to save the PDF (default: output.pdf in current directory).
        title: Optional title shown at the top of the first page.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        return f"ERROR: PyMuPDF not installed — {exc}"

    out = Path(output_path)
    if not out.is_absolute():
        out = Path(os.getcwd()) / out
    out.parent.mkdir(parents=True, exist_ok=True)

    # Detect primary direction: if >30% Arabic chars → RTL
    arabic_chars = sum(1 for c in text if "؀" <= c <= "ۿ")
    is_rtl = arabic_chars / max(len(text), 1) > 0.3
    direction = "rtl" if is_rtl else "ltr"
    text_align = "right" if is_rtl else "left"

    safe_title = html.escape(title) if title else ""
    pages_text = _split_pages(text)

    doc = fitz.open()
    for page_num, chunk in enumerate(pages_text):
        page = doc.new_page(width=595, height=842)
        safe_body = html.escape(chunk)
        title_html = (
            f'<h2 style="font-size:14pt; color:#2c3e50; margin-bottom:8pt;">{safe_title}</h2>'
            if safe_title and page_num == 0
            else ""
        )
        html_content = (
            f'<div dir="{direction}" style="'
            f'font-family: sans-serif; font-size: 11pt; line-height: 1.6; '
            f'text-align: {text_align};">'
            f"{title_html}"
            f'<p style="white-space: pre-wrap;">{safe_body}</p>'
            f"</div>"
        )
        page.insert_htmlbox(fitz.Rect(40, 40, 555, 802), html_content)

    doc.save(str(out))
    doc.close()
    return str(out)


__all__ = ["create_pdf"]

#!/usr/bin/env python3
"""Download a free Arabic book, embed it, and seed Qdrant.

This script uses the existing project stack:
  - Embeddings: BAAI/bge-m3
  - Vector DB : Qdrant (from .env)
  - Retrieval : semantic/hybrid via existing retriever
  - Answering : grounded answer with trust layer

Usage:
  uv run python scripts/seed_free_arabic_book_qdrant.py
  uv run python scripts/seed_free_arabic_book_qdrant.py --ask "ما موضوع الكتاب؟" --answer
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BOOKS_DIR = ROOT / "data" / "samples" / "books"
GUTENDEX_API = "https://gutendex.com/books"


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _arabic_ratio(text: str) -> float:
    if not text:
        return 0.0
    arabic = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
    letters = sum(1 for ch in text if ch.isalpha())
    if letters == 0:
        return 0.0
    return arabic / letters


def _pick_book(query: str = "") -> dict:
    params = {"languages": "ar"}
    if query.strip():
        params["search"] = query.strip()
    response = requests.get(GUTENDEX_API, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results") or []
    for item in results:
        formats = item.get("formats") or {}
        text_url = (
            formats.get("text/plain; charset=utf-8")
            or formats.get("text/plain")
            or formats.get("text/plain; charset=us-ascii")
        )
        if not text_url:
            continue
        try:
            preview_resp = requests.get(text_url, timeout=25)
            preview_resp.raise_for_status()
            preview = _clean_text(preview_resp.text[:20000])
        except Exception:
            continue
        if _arabic_ratio(preview) < 0.25:
            continue
        return {
            "id": item.get("id"),
            "title": item.get("title") or "Arabic book",
            "authors": ", ".join(a.get("name", "") for a in (item.get("authors") or [])),
            "text_url": text_url,
        }
    raise RuntimeError("No Arabic book with plain text download URL found from Gutendex.")


def _download_book_text(book: dict) -> Path:
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", (book["title"] or "arabic_book")).strip("_").lower()
    target = BOOKS_DIR / f"gutendex_{book['id']}_{slug[:80]}.txt"
    if target.is_file() and target.stat().st_size > 1000:
        return target
    response = requests.get(book["text_url"], timeout=45)
    response.raise_for_status()
    text = _clean_text(response.text)
    if len(text) < 1000:
        raise RuntimeError("Downloaded text is too short; choose another book/query.")
    target.write_text(text, encoding="utf-8")
    return target


def _build_index_from_text(book_file: Path, title: str, authors: str):
    from chunking.factory import chunk_document
    from embeddings.factory import embed_documents
    from ocr.base import OCRDocument, OCRPage
    from retriever.semantic import DocumentIndex
    from vectorstore.factory import get_vector_store

    text = book_file.read_text(encoding="utf-8")
    # Create pseudo-pages so existing chunking/retrieval pipeline can run unchanged.
    page_size = 3600
    pages: list[OCRPage] = []
    for i in range(0, len(text), page_size):
        body = text[i : i + page_size].strip()
        if body:
            pages.append(
                OCRPage(
                    page_number=(i // page_size) + 1,
                    text=body,
                    metadata={"source": "gutendex"},
                )
            )
    if not pages:
        raise RuntimeError("No text pages created from downloaded book.")

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    filename = f"{title}.txt"
    ocr_doc = OCRDocument(
        filename=filename,
        fingerprint=digest,
        pages=pages,
        engine="digital_text",
        metadata={"book_title": title, "authors": authors},
    )
    chunks = chunk_document(ocr_doc)
    if not chunks:
        raise RuntimeError("Chunking produced no text chunks.")
    embeddings = embed_documents([chunk.text for chunk in chunks])
    from retriever.knowledge import knowledge_collection_name

    collection_name = knowledge_collection_name()
    store = get_vector_store(collection_name=collection_name)
    store.add_chunks(chunks, embeddings, filename=filename)

    return DocumentIndex(
        fingerprint=digest,
        filename=filename,
        ocr=ocr_doc,
        chunks=chunks,
        store=store,
        metadata={
            "book_title": title,
            "authors": authors,
            "collection_name": collection_name,
            "page_count": len(pages),
            "chunk_count": len(chunks),
            "source_file": str(book_file),
        },
    )


def _retrieve(index, question: str, top_k: int = 5) -> None:
    from retriever.semantic import retrieve

    hits = retrieve(index, question, top_k=top_k)
    print(f"\nRetrieve: {question}")
    if not hits:
        print("  (no hits)")
        return
    for i, hit in enumerate(hits, start=1):
        snippet = hit.text.replace("\n", " ")[:170]
        print(f"  [{i}] score={hit.score:.3f} page={hit.page_start} {snippet}…")


def _answer(index, question: str) -> None:
    from agent.document_qa import answer_document_question_sync

    print(f"\nGrounded answer: {question}")
    answer = answer_document_question_sync(index, question)
    print(answer)


def main() -> int:
    from config.settings import get_settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", default="", help="Optional Arabic/English search keyword.")
    parser.add_argument("--ask", action="append", default=[], help="Question to test retrieval.")
    parser.add_argument(
        "--answer",
        action="store_true",
        help="Also generate grounded LLM answer (requires GROQ_API_KEY).",
    )
    args = parser.parse_args()

    settings = get_settings()
    if settings.vectorstore_backend != "qdrant":
        print(
            "Warning: VECTORSTORE_BACKEND is not 'qdrant'. "
            "Set VECTORSTORE_BACKEND=qdrant in .env for Qdrant seeding."
        )

    book = _pick_book(args.search)
    book_file = _download_book_text(book)
    index = _build_index_from_text(book_file, book["title"], book["authors"])

    print("Seed complete:")
    print(f"  title       : {book['title']}")
    print(f"  authors     : {book['authors'] or 'Unknown'}")
    print(f"  source      : {book_file}")
    print(f"  vector DB   : {settings.vectorstore_backend}")
    print(f"  qdrant URL  : {settings.qdrant_url}")
    print(f"  collection  : {index.metadata['collection_name']}")
    print(f"  pages/chunks: {index.metadata['page_count']} / {index.metadata['chunk_count']}")
    print(f"  store count : {index.store.count()}")

    questions = args.ask or [
        "ما موضوع هذا الكتاب؟",
        "من هو المؤلف؟",
    ]
    for q in questions:
        _retrieve(index, q, top_k=5)
        if args.answer:
            _answer(index, q)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


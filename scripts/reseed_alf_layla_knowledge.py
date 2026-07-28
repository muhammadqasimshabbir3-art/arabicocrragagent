#!/usr/bin/env python3
"""Rebuild wathiqa_knowledge with real Arabic Alf Layla OCR text.

Hardened against Qdrant cloud timeouts (small batches + retries + resume).

Recommended (faster, enough for topic questions):

  cd /home/engmatix/qasim_ai/ArabicOCRRAGAgent
  PYTHONUNBUFFERED=1 uv run python scripts/reseed_alf_layla_knowledge.py \\
    --chars 120000 --batch 8 --test-answer

If it times out mid-way, resume without deleting:

  PYTHONUNBUFFERED=1 uv run python scripts/reseed_alf_layla_knowledge.py \\
    --chars 120000 --batch 8 --resume --test-answer
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OCR_TXT = ROOT / "data" / "samples" / "books" / "alf_layla_part1_ocr.txt"
OCR_URL = (
    "https://archive.org/download/20211219_20211219_0955/"
    "%D9%83%D8%AA%D8%A7%D8%A8%20%D8%A3%D9%84%D9%81%20%D9%84%D9%8A%D9%84%D8%A9%20"
    "%D9%88%D9%84%D9%8A%D9%84%D8%A9%20%D8%A7%D9%84%D8%AC%D8%B2%D8%A1%20%D8%A7%D9%84%D8%A3%D9%88%D9%84"
    "%20-%20%D8%B1%D9%88%D8%A7%D9%8A%D8%A7%D8%AA%D9%8A_djvu.txt"
)


def log(*args: object) -> None:
    print(*args, flush=True)


def ensure_ocr_text() -> Path:
    if OCR_TXT.is_file() and OCR_TXT.stat().st_size > 100_000:
        return OCR_TXT
    import urllib.request

    OCR_TXT.parent.mkdir(parents=True, exist_ok=True)
    log("Downloading Arabic OCR text from Internet Archive…")
    urllib.request.urlretrieve(OCR_URL, OCR_TXT)
    log("Saved", OCR_TXT, "bytes=", OCR_TXT.stat().st_size)
    return OCR_TXT


def upsert_with_retries(store, part, vectors, filename: str, *, attempts: int = 6) -> None:
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            store.add_chunks(part, vectors, filename=filename)
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            wait = min(45, 2**attempt)
            log(f"  upsert retry {attempt}/{attempts} after error: {exc}")
            log(f"  sleeping {wait}s…")
            time.sleep(wait)
            # Recreate client connection on next call by rebuilding store wrapper.
            try:
                from vectorstore.factory import get_vector_store

                store_ref = get_vector_store(collection_name=store.collection_name)
                store._client = store_ref._client  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
    assert last is not None
    raise last


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chars",
        type=int,
        default=120_000,
        help="How many characters to index from the start (default 120000).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Embedding/upsert batch size (default 8 for cloud stability).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Do not delete the collection; re-upsert (safe/idempotent).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.8,
        help="Pause seconds between batches (default 0.8).",
    )
    parser.add_argument(
        "--test-answer",
        action="store_true",
        help="After seeding, ask: ما موضوع ألف ليلة وليلة؟",
    )
    args = parser.parse_args()

    from config.settings import get_settings

    get_settings.cache_clear()
    from chunking.factory import chunk_document
    from embeddings.factory import embed_documents
    from ocr.base import OCRDocument, OCRPage
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    from retriever.knowledge import knowledge_collection_name, retrieve_from_knowledge
    from vectorstore.factory import get_vector_store

    settings = get_settings()
    collection = knowledge_collection_name()
    src = ensure_ocr_text()
    full = src.read_text(encoding="utf-8", errors="replace")
    full = full.replace("\r\n", "\n").replace("\r", "\n")
    full = re.sub(r"\n{3,}", "\n\n", full).strip()
    text = full[: max(20_000, args.chars)]
    log(f"Using chars={len(text)} / full={len(full)} collection={collection}")

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=180)
    exists = client.collection_exists(collection)
    if exists and not args.resume:
        log("Deleting old collection…")
        client.delete_collection(collection)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=collection,
            vectors_config=qmodels.VectorParams(
                size=settings.embedding_dimensions,
                distance=qmodels.Distance.COSINE,
            ),
        )
        log("Created empty collection")
    else:
        info = client.get_collection(collection)
        log(f"Resuming into existing collection (points={info.points_count})")

    page_size = 3600
    pages: list[OCRPage] = []
    for i in range(0, len(text), page_size):
        body = text[i : i + page_size].strip()
        if body:
            pages.append(
                OCRPage(
                    page_number=len(pages) + 1,
                    text=body,
                    metadata={"source": "archive_ocr"},
                )
            )
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    filename = "alf_layla_wa_layla_part1.txt"
    ocr_doc = OCRDocument(
        filename=filename,
        fingerprint=digest,
        pages=pages,
        engine="digital_text",
        metadata={"title": "ألف ليلة وليلة"},
    )
    chunks = chunk_document(ocr_doc)
    log("pages", len(pages), "chunks", len(chunks))

    store = get_vector_store(collection_name=collection)
    batch = max(4, args.batch)
    total = 0
    for start in range(0, len(chunks), batch):
        part = chunks[start : start + batch]
        vectors = embed_documents([c.text for c in part])
        upsert_with_retries(store, part, vectors, filename)
        total += len(part)
        try:
            count = store.count()
        except Exception as exc:  # noqa: BLE001
            count = f"(count failed: {exc})"
        log(f"upserted {total}/{len(chunks)} store_count={count}")
        if args.sleep > 0:
            time.sleep(args.sleep)

    log("DONE store_count=", store.count())

    q = "ما موضوع ألف ليلة وليلة؟"
    hits = retrieve_from_knowledge(q, top_k=5)
    log("\nRetrieve test:")
    for i, hit in enumerate(hits, 1):
        snip = hit.text.replace("\n", " ")[:220]
        log(f"[{i}] score={hit.score:.3f} page={hit.page_start} {snip}…")

    if args.test_answer:
        from agent.document_qa import answer_knowledge_question_sync

        log("\n=== Grounded answer ===")
        log(answer_knowledge_question_sync(q))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

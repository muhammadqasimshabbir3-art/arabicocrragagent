#!/usr/bin/env python3
"""Seed a free Arabic sample PDF into the real (persistent) Chroma vector DB.

Stack used here (all free / open source except the optional Groq LLM answer):

  Vector DB   → Chroma (Apache-2.0) on disk at data/chroma
                https://github.com/chroma-core/chroma
  Embeddings  → BAAI/bge-m3 (MIT) via sentence-transformers
                https://huggingface.co/BAAI/bge-m3
  OCR         → Not needed for this digital PDF (text layer).
                Scanned docs use Qari-OCR (Apache-2.0 weights on HF):
                NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct
  Sample doc  → MIT sample Arabic circular authored in-repo
                (scripts/make_arabic_sample_pdf.py)

Usage (from repo root):
    uv run python scripts/seed_arabic_sample.py
    uv run python scripts/seed_arabic_sample.py --ask "ما مدة الإجازة السنوية؟"
    uv run python scripts/seed_arabic_sample.py --ask "What is the annual leave duration?"

Requires embedding model on disk or network access for first download:
    uv run python scripts/download_embedding_model.py
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SAMPLE_PDF = ROOT / "data" / "samples" / "arabic_admin_circular.pdf"
DEFAULT_QUESTIONS = (
    "ما مدة الإجازة السنوية للموظف؟",
    "What is the annual leave duration?",
    "متى يبدأ سريان هذا التعميم؟",
)


def _ensure_sample_pdf() -> Path:
    if SAMPLE_PDF.is_file() and SAMPLE_PDF.stat().st_size > 500:
        return SAMPLE_PDF
    maker = ROOT / "scripts" / "make_arabic_sample_pdf.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_arabic_sample_pdf", maker)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {maker}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.write_arabic_sample_pdf(SAMPLE_PDF)


def _index_pdf(pdf_path: Path, *, into_knowledge: bool = True):
    from config.settings import get_settings
    from retriever.semantic import get_or_build_index, upsert_index_into_knowledge

    settings = get_settings()
    payload = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
    index = get_or_build_index(
        payload,
        filename=pdf_path.name,
        mime_type="application/pdf",
    )
    persist = settings.vectorstore_persist_dir or "(in-memory)"
    print("── Indexed ──────────────────────────────────────────")
    print(f"  file          : {pdf_path}")
    print(f"  fingerprint   : {index.fingerprint[:16]}…")
    print(f"  OCR engine    : {index.ocr.engine}  (digital = no Qari needed)")
    print(f"  pages/chunks  : {index.metadata.get('page_count')} / {len(index.chunks)}")
    print(f"  vector DB     : {settings.vectorstore_backend} @ {persist}")
    print(f"  collection    : doc_{index.fingerprint[:16]}")
    print(f"  store count   : {index.store.count()}")
    print(f"  embeddings    : BAAI/bge-m3 (1024-d)")
    if into_knowledge:
        kb_count = upsert_index_into_knowledge(index)
        print(f"  knowledge DB  : {settings.knowledge_collection} ({kb_count} points)")
    return index


def _retrieve(index, question: str, top_k: int = 3) -> None:
    from retriever.semantic import retrieve

    hits = retrieve(index, question, top_k=top_k)
    print(f"\n── Retrieve: {question!r} ─────────────────")
    if not hits:
        print("  (no hits)")
        return
    for i, hit in enumerate(hits, start=1):
        snippet = hit.text.replace("\n", " ")[:160]
        print(f"  [{i}] score={hit.score:.3f} page={hit.page_start}  {snippet}…")


def _answer(index, question: str) -> None:
    from agent.document_qa import answer_document_question_sync

    print(f"\n── Grounded answer: {question!r} ──────────")
    try:
        answer = answer_document_question_sync(index, question)
    except Exception as exc:  # noqa: BLE001
        print(f"  Answer skipped (LLM unavailable): {exc}")
        print("  Set GROQ_API_KEY in .env to enable grounded answers.")
        return
    print(answer)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ask",
        action="append",
        default=[],
        help="Question to retrieve/answer (repeatable). Defaults to sample AR+EN questions.",
    )
    parser.add_argument(
        "--answer",
        action="store_true",
        help="Also call the Groq LLM for a grounded answer (needs GROQ_API_KEY).",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Optional path to another Arabic PDF to index instead of the sample.",
    )
    parser.add_argument(
        "--no-knowledge",
        action="store_true",
        help="Do not also upsert into the shared KNOWLEDGE_COLLECTION.",
    )
    args = parser.parse_args()

    pdf_path = args.pdf.resolve() if args.pdf else _ensure_sample_pdf()
    if not pdf_path.is_file():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    print("Stack (free / open source):")
    print("  • Vector DB  : configured via VECTORSTORE_BACKEND (chroma/faiss/qdrant)")
    print("  • Embeddings : BAAI/bge-m3 (MIT) — multilingual Arabic+English")
    print("  • OCR        : Qari-OCR for scans; this sample uses digital text")
    print()

    index = _index_pdf(pdf_path, into_knowledge=not args.no_knowledge)
    questions = args.ask or list(DEFAULT_QUESTIONS)
    for question in questions:
        _retrieve(index, question)
        if args.answer:
            _answer(index, question)

    print("\nDone. Upload the same PDF in the UI to ask interactively:")
    print(f"  {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

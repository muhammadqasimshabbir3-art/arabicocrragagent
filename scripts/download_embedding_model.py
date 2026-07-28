#!/usr/bin/env python3
"""Download the configured embedding model into models/ for local semantic search.

Default: BAAI/bge-m3 (multilingual Arabic + English).

Usage (from repo root):
    python scripts/download_embedding_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from agent.embeddings import MODEL_ID, get_embedding_model, model_dir

    target = model_dir()
    print(f"Downloading {MODEL_ID}")
    print(f"Saving to   {target}")
    model = get_embedding_model()
    dims = getattr(model, "get_embedding_dimension", None)
    if callable(dims):
        dim_count = dims()
    else:
        dim_count = model.get_sentence_embedding_dimension()
    probe = model.encode(
        "ما هي سياسة الاسترجاع؟",
        normalize_embeddings=True,
    )
    print(f"Ready. dimensions={dim_count}, probe_len={len(probe)}")
    print("OCR model downloads automatically on first scanned-document ingest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Ensure BGE-M3 embeddings are present under src/models/.

Usage (from repo root):
    uv run python scripts/ensure_models.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from core.config.settings import get_settings
    from subagents.embeddings.bge_m3 import model_dir as embedding_model_dir

    settings = get_settings()
    settings.models_dir.mkdir(parents=True, exist_ok=True)

    emb_dir = embedding_model_dir()
    emb_ok = emb_dir.is_dir() and (
        (emb_dir / "config.json").is_file() or (emb_dir / "modules.json").is_file()
    )
    print(f"BGE-M3: {'OK' if emb_ok else 'MISSING'} ({emb_dir})")
    if emb_ok:
        print("Embedding model is present under src/models/.")
        return 0

    cmd = [sys.executable, str(ROOT / "scripts" / "download_embedding_model.py")]
    print(f"→ {' '.join(cmd)}")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

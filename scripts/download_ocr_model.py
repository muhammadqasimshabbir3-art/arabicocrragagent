#!/usr/bin/env python3
"""Download the configured OCR model into models/ for offline restarts.

Default: NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct

Usage (from repo root):
    python scripts/download_ocr_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from config.settings import get_settings
    from ocr.qari import _load_qari, _model_dir

    settings = get_settings()
    print(f"Downloading OCR engine={settings.ocr_engine}")
    print(f"Model ID: {settings.ocr_model_id}")
    print(f"Saving to: {_model_dir()}")
    _load_qari()
    print("OCR model ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

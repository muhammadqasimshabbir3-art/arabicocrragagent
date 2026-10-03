# Backend models (`src/models/`)

Canonical weights directory for the Railway / LangGraph backend.

| Model | Path | Size | Required |
|-------|------|------|----------|
| **BAAI/bge-m3** | `src/models/bge-m3/` | ~2.2 GB | Yes (local embeddings) |

```bash
# from repo root
uv run python scripts/ensure_models.py
```

Env overrides:

```bash
MODELS_DIR=/deps/ArabicOCRRAGAgent/src/models
EMBEDDING_MODEL_PATH=/deps/ArabicOCRRAGAgent/src/models/bge-m3
```

Large `*.safetensors` files are gitignored — prefetch onto a Railway volume or local disk before serving traffic. There is **no** OCR model; text extraction is digital-only.

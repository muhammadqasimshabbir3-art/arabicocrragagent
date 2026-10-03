# Troubleshooting

## Backend offline / CORS

- Local: `./start.sh both` — API `:2024`, UI `:5173`.
- Production: absolute `VITE_LANGGRAPH_API_URL`; Railway `CORS_ALLOW_ORIGINS` includes Vercel.
- No `http.cors` block in `langgraph.json`.

## Scanned PDF / image upload fails

Expected. This agent extracts **digital text only**. Use a text-layer PDF, TXT, or DOCX.

## Missing embeddings

```bash
uv run python scripts/ensure_models.py
# must create src/models/bge-m3
```

On Railway, mount a volume at `/deps/ArabicOCRRAGAgent/src` or set `MODELS_DIR` / `EMBEDDING_MODEL_PATH`.

## Empty or weak answers

- Seed `KNOWLEDGE_COLLECTION`; confirm `VECTORSTORE_BACKEND`.
- Wait for summarize/index before asking about uploads.

## Groq errors

- Missing `GROQ_API_KEY` or rate limits.
- **Retired models:** `llama-3.1-8b-instant` / `llama-3.3-70b-versatile` shut down 2026-08-16. Set `GROQ_MODEL=openai/gpt-oss-120b` (accuracy) or `openai/gpt-oss-20b` (speed).
- **403 / “Access denied” / Cloudflare 1010:** network or account block from this host — try another network/VPN off, or a fresh key at [console.groq.com](https://console.groq.com).

## Railway Docker build: `langgraph-api` / `grpcio` unsatisfiable

Error looks like:

```text
Because langgraph-api==0.10.0 depends on grpcio>=1.80.0,<1.81.0 and grpcio>=1.81.0,<1.82.0
... requirements are unsatisfiable
```

**Cause:** `langgraph-api` must not be listed in `src/pyproject.toml` dependencies. The base image `langchain/langgraph-api:3.12` already provides the API; reinstalling it under `/api/constraints.txt` fights itself on `grpcio`.

**Fix:** remove `langgraph-api==…` from `[project].dependencies`, commit, redeploy. Local `langgraph dev` still uses `langgraph-cli[inmem]` from the `dev` dependency group.

## Railway healthcheck

- Port **8000** (`PORT=8000`).
- Dockerfile is `src/Dockerfile` (Root Directory = `src`).

## Tests

```bash
uv sync --group dev
uv run ruff check src tests
uv run pytest tests/unit_tests -q
```

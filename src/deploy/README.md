# Backend deploy (Railway) — `src/`

Set the Railway service **Root Directory** to **`src`**.

| File in `src/` | Role |
|----------------|------|
| `Dockerfile` | LangGraph API image |
| `railway.json` | Builder + `/ok` healthcheck |
| `langgraph.json` | Graph: `agent/graph.py:graph` |
| `pyproject.toml` / `uv.lock` | App deps only — **do not** pin `langgraph-api` (image provides it) |
| `models/` | BGE-M3 weights |
| `data/` | Chroma persistence |
| `deploy/.env.railway.example` | Env template |

## Steps

1. Railway → New Project → GitHub repo.
2. Settings → **Root Directory** = `src`.
3. Deploy uses `Dockerfile` + `railway.json` automatically.
4. Port **8000** (`PORT=8000`).
5. Paste variables from [`.env.railway.example`](./.env.railway.example)  
   (`GROQ_MODEL=openai/gpt-oss-120b` recommended for Arabic RAG accuracy).
6. Volume (recommended): `/deps/ArabicOCRRAGAgent` (covers `models/` + `data/`).
7. Prefetch: from repo root → `uv --directory src run python ../scripts/ensure_models.py`
8. `curl https://YOUR-SERVICE.up.railway.app/ok`
9. Set `CORS_ALLOW_ORIGINS` to your Vercel URL.

## Frontend (Vercel)

- Root Directory = **`frontend`**
- Config: `frontend/vercel.json`
- Env: `.env.vercel.example` at repo root / `frontend/.env.example`

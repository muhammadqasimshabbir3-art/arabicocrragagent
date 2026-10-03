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
5. **New → Database → PostgreSQL** and **New → Database → Redis** (same project).
6. Agent variables — LangGraph needs these exact names:

```bash
DATABASE_URI=${{Postgres.DATABASE_URL}}
REDIS_URI=${{Redis.REDIS_URL}}
```

7. Paste the rest from [`.env.railway.example`](./.env.railway.example)  
   (`GROQ_MODEL=openai/gpt-oss-120b` recommended).
8. Volume (recommended): `/deps/ArabicOCRRAGAgent` (covers `models/` + `data/`).
9. Prefetch: from repo root → `uv --directory src run python ../scripts/ensure_models.py`
10. `curl https://arabicocrragagent-production.up.railway.app/ok`
11. `CORS_ALLOW_ORIGINS=https://arabicocrragagent.vercel.app,https://smith.langchain.com` → restart.

## Frontend (Vercel)

- Root Directory = **`frontend`**
- Config: `frontend/vercel.json`
- Env: `.env.vercel.example` at repo root / `frontend/.env.example`

# Deployment — `src/` on Railway + `frontend/` on Vercel

| Piece | Root Directory | Key files |
|-------|----------------|-----------|
| Backend | **`src`** | `Dockerfile`, `railway.json`, `langgraph.json`, `pyproject.toml` |
| Frontend | **`frontend`** | `vercel.json`, Vite app |

Local first (dev or prod-like):

```bash
cp src/.env.example .env   # set GROQ_API_KEY
./setup.sh --skip-run
./start.sh both            # local dev (API + Vite HMR)
# or
./start.sh prod            # prod-like: build UI + API + vite preview
```

Scripts keep local `VITE_LANGGRAPH_API_URL` + CORS so frontend ↔ backend connect.

---

## Railway (backend)

1. New project → GitHub repo.
2. Service **Settings → Root Directory** = `src`.
3. Builder picks up `src/Dockerfile` + `src/railway.json`.
4. Port **8000**; `PORT=8000`.
5. Variables from [`src/deploy/.env.railway.example`](../src/deploy/.env.railway.example).
6. Volume at `/deps/ArabicOCRRAGAgent` (models + chroma).
7. `curl https://YOUR-SERVICE.up.railway.app/ok`
8. After Vercel: set `CORS_ALLOW_ORIGINS` to the Vercel origin.

```bash
GROQ_API_KEY=...
GROQ_MODEL=openai/gpt-oss-120b
MODELS_DIR=/deps/ArabicOCRRAGAgent/models
VECTORSTORE_PERSIST_DIR=/deps/ArabicOCRRAGAgent/data/chroma
OCR_ENGINE=digital
```

Use `openai/gpt-oss-20b` only if you need lower cost/latency. Do **not** use retired `llama-3.1-8b-instant` / `llama-3.3-70b-versatile`.

**Do not** add `langgraph-api` to `src/pyproject.toml` — the Docker base image already includes it. Pinning it breaks Railway builds (`grpcio` conflict). See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md).

---

## Vercel (frontend)

1. Import repo → **Root Directory** = `frontend`.
2. Uses [`frontend/vercel.json`](../frontend/vercel.json).
3. Env from [`.env.vercel.example`](../.env.vercel.example):

```bash
VITE_LANGGRAPH_API_URL=https://YOUR-SERVICE.up.railway.app
VITE_LANGGRAPH_ASSISTANT_ID=agent
```

4. Redeploy after any `VITE_*` change.

---

## Order

1. Local smoke: `./setup.sh` → `./start.sh both`
2. Push → Railway (`src`) → `/ok`
3. Vercel (`frontend`) → absolute Railway URL
4. CORS on Railway → restart

More: [src/deploy/README.md](../src/deploy/README.md) · [TROUBLESHOOTING.md](./TROUBLESHOOTING.md).

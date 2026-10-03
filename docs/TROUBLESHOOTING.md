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

## Railway: `Application startup failed` / `NoneType.encode` (psycopg)

Log contains:

```text
Using langgraph_runtime_postgres
AttributeError: 'NoneType' object has no attribute 'encode'
```

**Cause:** the Docker image `langchain/langgraph-api` needs a Postgres URI. `DATABASE_URI` is missing (or named wrong — Railway’s default is often `DATABASE_URL`).

**Fix on Railway (agent service):**

1. Add **PostgreSQL** and **Redis** to the **same project**.
2. Open the **agent** service → Variables → **Add variable → Add reference** (do not type a fake string).
3. Pick your Postgres service’s `DATABASE_URL` and Redis’s `REDIS_URL`.
4. Either name them for LangGraph:

```bash
DATABASE_URI=<reference to Postgres DATABASE_URL>
REDIS_URI=<reference to Redis REDIS_URL>
```

   or keep Railway defaults `DATABASE_URL` / `REDIS_URL` — our `deploy/railway-entrypoint.sh` aliases them.
5. Confirm the resolved value is a real `postgresql://…` URL (not empty, not the literal `${{…}}` text).
6. Redeploy. Logs should show `LangGraph env OK: DATABASE_URI set, REDIS_URI set` then `/ok` works.

**Common mistake:** Postgres exists in the project but is **not referenced** on the agent service → `DATABASE_URI` stays empty → same crash. The service name inside `${{Name.DATABASE_URL}}` must match the Postgres service name in Railway (often `Postgres`, sometimes `PostgreSQL`).

Also keep `CORS_ALLOW_ORIGINS=https://arabicocrragagent.vercel.app,...`.

## Railway: `License verification failed` / LangSmith `403 Forbidden`

Log contains:

```text
Error refreshing LangSmith access: Client error '403 Forbidden'
License verification failed
```

**Meaning:** the `langchain/langgraph-api` Docker image is a **licensed Agent Server**. It will not start until LangSmith accepts your key.

**Fix:**

1. Create/get a key at [smith.langchain.com](https://smith.langchain.com) → Settings → API Keys.
2. On Railway agent Variables set:

```bash
LANGSMITH_API_KEY=lsv2_pt_...
```

3. Your LangSmith org must allow **LangGraph / Deployments** for that key (free tracing-only keys often get **403**).
4. For paid self-host production, set `LANGGRAPH_CLOUD_LICENSE_KEY` instead/in addition.
5. Remove broken overrides if present: `LANGSMITH_ENDPOINT`, `LANGCHAIN_ENDPOINT` (unless you intentionally use EU/self-hosted LangSmith).

Without a valid key, the container exits even when Postgres/Redis are correct.

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

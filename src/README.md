# Wathiqa backend (`src/`)

Self-contained LangGraph backend for Railway and local `uv` / Docker.

## Deploy files (this folder)

| File | Purpose |
|------|---------|
| `pyproject.toml` / `uv.lock` | Python project |
| `langgraph.json` | Graph entry `./agent/graph.py:graph` |
| `Dockerfile` | Production image |
| `railway.json` | Railway build/deploy |
| `.env.example` | Backend env template |
| `deploy/` | Railway env + notes |
| `models/` | BGE-M3 |
| `data/` | Chroma |

## Local (from monorepo root)

```bash
cp src/.env.example .env   # set GROQ_API_KEY
./setup.sh
./start.sh both
```

Or only backend:

```bash
cd src
uv sync
uv run langgraph dev --port 2024
```

## Graph inputs (search source)

| Field | Effect |
|-------|--------|
| `use_web_search` | Force `web_search` route (Search Internet toggle) |
| `use_knowledge_base` | Force KB path when no upload (Database toggle) |

Frontend maps UI `search_source` (`auto` \| `database` \| `web`) → these flags.

## Railway

Root Directory = **`src`**. See [deploy/README.md](./deploy/README.md).

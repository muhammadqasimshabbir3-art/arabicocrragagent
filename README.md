# Arabic Document Intelligence Agent

**وثيقة (Wathiqa)** — LangGraph agent for Arabic document indexing and grounded RAG with citations.

Digital text from text-layer PDF / TXT / DOCX → BGE-M3 → Chroma/Qdrant → grounded answers with citations.

| | |
|---|---|
| **Author** | Muhammad Qasim Shabbir |
| **Email** | [muhammadqasimshabbir3@gmail.com](mailto:muhammadqasimshabbir3@gmail.com) |
| **GitHub** | [muhammadqasimshabbir3-art](https://github.com/muhammadqasimshabbir3-art) |
| **LinkedIn** | [muhammad-qasim-shabbir](https://www.linkedin.com/in/muhammad-qasim-shabbir-b93b25355/) |
| **Zindi** | One of Africa’s / the world’s top data science & AI competition platforms — [MuhammadQasimShabbeer](https://zindi.world/users/MuhammadQasimShabbeer) · rank **#13** · **10,182** pts · best **#12** · [portfolio](https://zindi.world/users/MuhammadQasimShabbeer/competitions/portfolio) |
| **Version** | 1.0.0 |
| **License** | MIT |

---

## Deploy layout

| Piece | Folder | Platform | Config |
|-------|--------|----------|--------|
| **Backend** | [`src/`](src/README.md) | Railway | `Dockerfile` · `railway.json` · `langgraph.json` · `pyproject.toml` |
| **Frontend** | [`frontend/`](frontend/) | Vercel | `vercel.json` |

```text
frontend/          → Vercel  (Root Directory = frontend)
src/               → Railway (Root Directory = src)
  agent/ core/ subagents/ tools/
  models/          BGE-M3
  data/            Chroma
  Dockerfile
  railway.json
  langgraph.json
  pyproject.toml
```

No neural OCR — text-layer documents only.

---

## Local & production-like runs

```bash
cp src/.env.example .env
# set GROQ_API_KEY

chmod +x setup.sh start.sh
./setup.sh --skip-run          # install backend (src/) + frontend
./start.sh both                # LOCAL DEV: API + React (connected)
```

| Command | Mode | What runs |
|---------|------|-----------|
| `./start.sh both` | **Local dev** | LangGraph `:2024` + Vite HMR `:5173` |
| `./start.sh prod` | **Prod-like local** | Build UI + API + `vite preview` |
| `./start.sh build` | Build only | `frontend/dist` for Vercel |
| `./start.sh docker` | Docker API | `docker compose up api` (`src/Dockerfile`) |
| `./start.sh stop` | Ops | Stop API / UI / Streamlit |

- API: http://127.0.0.1:2024  
- UI: http://127.0.0.1:5173  

Scripts auto-set local `VITE_LANGGRAPH_API_URL` and CORS so UI ↔ API connect.

### Search source (Search Internet / Database)

In the React UI, next to **Answer language**, choose where the agent looks:

| Option | Meaning |
|--------|---------|
| **Auto** | Default heuristics (document upload, KB keywords, or live-web intent) |
| **Database** | Explicitly query the seeded knowledge base (no file attached) |
| **Search Internet** | Explicit live web search (Google-style) for this question |

These map to graph inputs `use_knowledge_base` / `use_web_search`. Search Internet wins if both are set. See [AgentWorkflow.md](AgentWorkflow.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

**Cloud deploy:** Railway Root Directory = `src` · Vercel Root Directory = `frontend`  
(see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)).

Prefetch embeddings:

```bash
uv --directory src run python ../scripts/ensure_models.py
```

---

## Documentation

| Doc | Topic |
|-----|-------|
| [PLAN.md](PLAN.md) | Rebuild / fork checklist for another agent |
| [src/README.md](src/README.md) | Backend |
| [src/deploy/README.md](src/deploy/README.md) | Railway |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Full Railway + Vercel |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Env vars |
| [AgentWorkflow.md](AgentWorkflow.md) | Routes |

---

## Tech stack

| Layer | Tech |
|-------|------|
| Orchestration | LangGraph |
| LLM | Groq **`openai/gpt-oss-120b`** (default) |
| Text | Digital (`pypdf`) |
| Embeddings | BGE-M3 in `src/models/bge-m3` |
| Vectors | Chroma (`src/data/chroma`) / Qdrant |
| UI | React 19 + Vite |

### Groq chat model (free / developer)

Groq retired `llama-3.1-8b-instant` and `llama-3.3-70b-versatile` on **2026-08-16**. This agent defaults to a stronger production model for Arabic grounded Q&A:

| `GROQ_MODEL` | Role | When to use |
|--------------|------|-------------|
| **`openai/gpt-oss-120b`** | **Default** — highest accuracy among free/dev production models | Arabic RAG, citations, bilingual answers |
| `openai/gpt-oss-20b` | Faster / cheaper | High traffic, latency-sensitive |
| `qwen/qwen3.8-27b` | Multilingual preview | Extra Arabic experimentation (preview; may change) |

Set in `.env` (or Railway variables):

```bash
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-120b
```

---

## Deploy

1. Push to GitHub.
2. **Railway** — Root Directory `src` → vars from `src/deploy/.env.railway.example` → `/ok`.
3. **Vercel** — Root Directory `frontend` → `VITE_LANGGRAPH_API_URL` = Railway URL.
4. Railway `CORS_ALLOW_ORIGINS` = Vercel origin → restart.

---

## Testing

```bash
uv --directory src sync --group dev
uv --directory src run pytest ../tests/unit_tests -q
```

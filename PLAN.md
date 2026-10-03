# PLAN.md — Rebuild Wathiqa-style Arabic Document Intelligence Agent

Use this plan in **another agent / repo** to recreate the same product shape: LangGraph backend in `src/`, React UI in `frontend/`, Railway + Vercel deploy, ChatGPT-style Website/Database toggles, and author profile links.

---

## 1. Goal

Ship an Arabic document intelligence agent (**وثيقة / Wathiqa**) that:

1. Accepts text-layer PDF / TXT / DOCX (digital extract only — **no neural OCR**).
2. Indexes with BGE-M3 → Chroma (local) or Qdrant (prod).
3. Answers with grounded citations (retrieve → trust layer → answer + Sources).
4. Routes turns to: document QA, summarize, knowledge DB, live web search, or chat.
5. Exposes UI toggles: **Answer language** + **Search source** (Auto / Database / Website).
6. Shows author social links (GitHub, LinkedIn, Zindi + rank/portfolio).
7. Deploys: **Railway** (`src/`) + **Vercel** (`frontend/`).

---

## 2. Monorepo layout (required)

```text
repo/
├── frontend/                 # Vercel Root Directory = frontend
│   ├── vercel.json
│   ├── package.json
│   └── src/
│       ├── App.tsx
│       ├── components/       # AgentHeader, AgentConfigForm, AuthorProfile, …
│       ├── lib/
│       │   ├── agentClient.ts
│       │   ├── authorLinks.ts
│       │   ├── defaultSettings.ts
│       │   └── workflowSteps.ts
│       └── types.ts
├── src/                      # Railway Root Directory = src
│   ├── agent/graph.py        # LangGraph entry
│   ├── core/                 # config, routing, retriever, vectorstore, …
│   ├── subagents/            # document_qa, embeddings, query_planner, ocr/digital
│   ├── tools/                # web_search, files, …
│   ├── models/bge-m3/        # embedding weights
│   ├── data/                 # chroma persist (local / volume)
│   ├── deploy/.env.railway.example
│   ├── Dockerfile
│   ├── railway.json
│   ├── langgraph.json        # graph = ./agent/graph.py:graph
│   ├── pyproject.toml
│   └── .env.example
├── scripts/                  # ensure_models, seed_*, services.sh
├── tests/unit_tests/
├── docs/                     # ARCHITECTURE, CONFIGURATION, DEPLOYMENT, …
├── setup.sh / start.sh
├── README.md
├── AgentWorkflow.md
└── PLAN.md                   # this file
```

**Do not** put Railway/`langgraph.json`/`pyproject.toml` at monorepo root if Root Directory is `src`.

---

## 3. Product decisions (lock these)

| Decision | Choice |
|----------|--------|
| Text extraction | Digital only (`OCR_ENGINE=digital`, pypdf / loaders) |
| Embeddings | BGE-M3 under `src/models/bge-m3` |
| Default LLM | Groq `openai/gpt-oss-120b` (`GROQ_MODEL`) |
| Local vectors | Chroma in `src/data/chroma` |
| Prod vectors | Qdrant via `VECTORSTORE_BACKEND=qdrant` |
| Knowledge corpus | Collection `KNOWLEDGE_COLLECTION` (e.g. `wathiqa_knowledge`) |
| Frontends | React primary; Streamlit optional/legacy |
| CORS | Env `CORS_ALLOW_ORIGINS` (Vercel origin in prod) |

---

## 4. Graph & routing

### Nodes

```text
START → prepare_input → decision_agent
  ├── query_documents
  ├── summarize_document
  ├── query_planner → query_knowledge_base
  ├── web_search
  └── call_model
→ END
```

No separate `ingest_document` node — ingest runs inside query/summarize with cache.

### Explicit UI search source (required)

State fields:

- `use_web_search: bool`
- `use_knowledge_base: bool`
- `response_language: "ar" | "en"`

`pick_route(...)` priority:

1. If `use_web_search` → **`web_search`** (highest; even with a file attached).
2. Else if document attached → document path (query / summarize); heuristics may still pick web.
3. Else if `use_knowledge_base` → **`query_knowledge_base`** when KB non-empty, else chat.
4. Else heuristics (`wants_web_lookup` / `wants_knowledge_lookup` / chat).

Frontend mapping:

| UI `search_source` | Flags |
|--------------------|-------|
| `auto` | both false |
| `database` | `use_knowledge_base=true` |
| `web` | `use_web_search=true` |

Wire in `buildAgentInput()` so every run sends the flags.

---

## 5. Frontend checklist

1. **Composer toggles**
   - Answer language: العربية / English
   - Search source: Auto / Database / Website
2. **Upload** → auto-summarize path; follow-up questions use document mode.
3. **Stream** LangGraph runs via SDK (`VITE_LANGGRAPH_API_URL`).
4. **Author profiles** (see §7) in header icons + author panel (Zindi rank card).
5. Keep `frontend/vercel.json`; set Root Directory = `frontend` on Vercel.

Env:

```bash
VITE_LANGGRAPH_API_URL=https://YOUR-RAILWAY.up.railway.app
VITE_LANGGRAPH_ASSISTANT_ID=agent
```

---

## 6. Backend checklist

1. Package under `src/` with `uv` + `langgraph.json`.
2. Settings from env (`core/config/settings.py`).
3. Hybrid retrieve (BM25 + vectors); optional reranker.
4. Trust layer before citing sources.
5. Spelling Guardian optional (`ENABLE_SPELLING_CORRECTOR`).
6. Web tools for `web_search` route.
7. Dockerfile serves LangGraph API; health `/ok`.
8. Railway volume for models + chroma if used.
9. **Never** pin `langgraph-api` in `pyproject.toml` (base image provides it; pin → Railway `grpcio` resolve failure).

Copy vars from `src/deploy/.env.railway.example` (GROQ, MODELS_DIR, VECTORSTORE_*, OCR_ENGINE=digital, CORS).

---

## 7. Author / social links (copy into new agent)

Centralize in `frontend/src/lib/authorLinks.ts` and mirror in README:

| Link | URL |
|------|-----|
| GitHub | https://github.com/muhammadqasimshabbir3-art |
| LinkedIn | https://www.linkedin.com/in/muhammad-qasim-shabbir-b93b25355/ |
| Zindi profile | https://zindi.world/users/MuhammadQasimShabbeer |
| Zindi portfolio | https://zindi.world/users/MuhammadQasimShabbeer/competitions/portfolio |

Zindi stats to display:

- Current rank: **#13**
- Points: **10,182**
- Best rank: **#12**

UI placement:

- Compact GitHub / LinkedIn / Zindi icons in the topbar.
- Author panel under the desk rail: name, email, social chips, Zindi stats + portfolio link.

README table: Author, Email, GitHub, LinkedIn, Zindi (rank + portfolio).

---

## 8. Local scripts (parity)

```bash
cp src/.env.example .env   # GROQ_API_KEY=...
chmod +x setup.sh start.sh
./setup.sh --skip-run
./start.sh both            # API :2024 + Vite :5173
# or
./start.sh prod            # build + preview (prod-like)
```

Scripts must set local `VITE_LANGGRAPH_API_URL` and CORS so UI ↔ API connect.

Prefetch embeddings:

```bash
uv --directory src run python ../scripts/ensure_models.py
```

Tests:

```bash
uv --directory src sync --group dev
uv --directory src run pytest ../tests/unit_tests -q
```

Include routing tests for explicit Website/Database flags.

---

## 9. Cloud publish sequence

1. Push GitHub.
2. **Railway**: Root Directory = `src` → env from deploy example → `/ok`.
3. **Vercel**: Root Directory = `frontend` → `VITE_LANGGRAPH_API_URL` = Railway HTTPS.
4. Railway `CORS_ALLOW_ORIGINS` = Vercel origin → redeploy/restart.
5. Smoke: upload PDF, Auto ask, Database ask, Website ask, open social links.

Docs to keep in sync: `README.md`, `AgentWorkflow.md`, `docs/ARCHITECTURE.md`, `docs/CONFIGURATION.md`, `docs/DEPLOYMENT.md`, `src/README.md`.

---

## 10. Implementation order (for the other agent)

1. Scaffold monorepo folders + `pyproject.toml` / Vite app.
2. Implement graph nodes + `pick_route` with web/KB flags.
3. Digital loaders → chunk → embed → retrieve → trust → answer.
4. Knowledge path + web_search tools.
5. React composer: language + search source → `buildAgentInput`.
6. Author links + Zindi card.
7. `setup.sh` / `start.sh` + CORS/env wiring.
8. Docker + Railway + Vercel configs.
9. Unit tests (routing, config, sanitize).
10. README + PLAN + architecture docs.
11. Seed knowledge corpus (optional scripts).
12. Publish and verify CORS + toggles in production.

---

## 11. Out of scope / avoid

- Neural OCR models in the hot path (keep digital-only unless product changes).
- Putting backend package root outside `src/` when Railway Root is `src`.
- Shipping secrets in `VITE_*`.
- Forcing Database path when a file is attached (document path wins unless Website is on).
- Purple/cream AI-slop redesign; keep the existing ochre / Cairo / Amiri visual language if forking this UI.

---

## 12. Definition of done

- [ ] Local `./start.sh both` — UI online, API healthy.
- [ ] Upload → auto summary → follow-up with citations.
- [ ] Search source **Website** forces web route.
- [ ] Search source **Database** hits KB when seeded.
- [ ] Header + panel show GitHub, LinkedIn, Zindi (#13 / 10182 / best #12 / portfolio).
- [ ] Railway `/ok` + Vercel UI talk with CORS set.
- [ ] README lists the same author links.
- [ ] This PLAN.md kept with the repo for the next fork.

---

*Source project: Arabic OCR / RAG Agent (Wathiqa) — Muhammad Qasim Shabbir.*

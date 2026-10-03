# Configuration

Local: `cp src/.env.example .env` (or root `.env.example`).  
Railway: `src/deploy/.env.railway.example` (Root Directory = `src`).  
Vercel: `.env.vercel.example` (Root Directory = `frontend`).

## Core

| Variable | Default | Notes |
|----------|---------|-------|
| `GROQ_API_KEY` | — | Required for Groq |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Default chat model (highest free/dev accuracy for Arabic RAG). Alternatives: `openai/gpt-oss-20b` (faster), `qwen/qwen3.8-27b` (multilingual preview). Old Llama 3.1/3.3 IDs retired 2026-08-16. |
| `LLM_PROVIDER` | `groq` | `groq` \| `openai` \| `ollama` \| `gemini` |
| `LLM_TEMPERATURE` | `0` | |

## Text extraction

| Variable | Default | Notes |
|----------|---------|-------|
| `OCR_ENGINE` | `digital` | Digital text only (PDF text layer / TXT / DOCX) |
| `OCR_MAX_PAGES` | `120` | Loader page cap |
| `SUMMARY_MAX_CHARS` | `48000` | Max chars for summarizer |

Scanned / image-only documents are not supported.

## Embeddings & retrieval

| Variable | Default | Notes |
|----------|---------|-------|
| `EMBEDDING_MODEL_ID` | `BAAI/bge-m3` | |
| `MODELS_DIR` | `src/models` | Backend models root |
| `EMBEDDING_MODEL_PATH` | — | Optional override → `src/models/bge-m3` |
| `CHUNK_STRATEGY` | `page_aware` | Also recursive / layout_aware / semantic |
| `VECTORSTORE_BACKEND` | `chroma` | `faiss` \| `qdrant` |
| `VECTORSTORE_PERSIST_DIR` | `src/data/chroma` | Chroma path |
| `QDRANT_URL` / `QDRANT_API_KEY` | — | When using Qdrant |
| `KNOWLEDGE_COLLECTION` | `wathiqa_knowledge` | Seeded corpus |
| `RETRIEVAL_MODE` | `hybrid` | or `semantic` |
| `HYBRID_FUSION` | `rrf` | `weighted` for 0.7/0.3 rollback |
| `ENABLE_RERANKER` | `true` | Cross-encoder by default |
| `MAX_DOCUMENT_BYTES` | `52428800` | 50 MB |

```bash
uv run python scripts/ensure_models.py
# → src/models/bge-m3
```

## Ops

| Variable | Default | Notes |
|----------|---------|-------|
| `LOG_LEVEL` | `INFO` | |
| `LOG_FORMAT` | `text` | `json` on Railway |
| `VALIDATE_SETTINGS_ON_STARTUP` | off | `true` in production |
| `CORS_ALLOW_ORIGINS` | localhost + Studio | Comma-separated; no `http.cors` in `langgraph.json` |
| `PORT` | — | Set `8000` on Railway |

## Frontend (`VITE_*`)

| Variable | Notes |
|----------|-------|
| `VITE_LANGGRAPH_API_URL` | Absolute Railway HTTPS URL |
| `VITE_LANGGRAPH_ASSISTANT_ID` | Default `agent` |
| `VITE_STACK_OCR` | UI label, e.g. `Digital text (pypdf)` |

### Run inputs (React → LangGraph)

| Field | Values | Notes |
|-------|--------|-------|
| `response_language` | `ar` \| `en` | Forced answer language |
| `search_source` (UI) | `auto` \| `database` \| `web` | Mapped to flags below |
| `use_knowledge_base` | bool | Force Database / KB path |
| `use_web_search` | bool | Force Search Internet / live web search |

Do not ship secrets in `VITE_*`. Deploy: [DEPLOYMENT.md](./DEPLOYMENT.md).

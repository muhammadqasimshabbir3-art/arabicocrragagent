# Architecture

## Overview

Wathiqa is a LangGraph `StateGraph` that routes each turn to document QA, knowledge-base retrieval, web search, summarize, or general chat. Answers that cite documents are grounded via a trust layer over retrieved passages.

## Graph

```text
START → prepare_input → decision_agent
  ├── query_documents          upload + question → retrieve → trust → answer
  ├── summarize_document       upload / summarize flag → summary
  ├── query_planner → query_knowledge_base   seeded corpus (READ-only plan)
  ├── web_search               live / online intent
  └── call_model               general chat
→ END
```

There is **no** separate `ingest_document` node: upload turns load → digital extract → chunk → index inside `query_documents` / `summarize_document` (with index cache).

## Layers

| Package | Role |
|---------|------|
| `src/agent/` | Main LangGraph orchestrator (`graph.py`) |
| `src/subagents/` | Digital text extract, embeddings, reranker, document QA, planner |
| `src/core/` | Infra: routing, loaders, preprocess, chunking, retriever, vectorstore, llm, config |
| `src/tools/` | LLM-callable tools |
| `src/models/` | BGE-M3 embedding weights |
| `frontend/` | React streaming UI (Vercel) |

## Data flow (upload QA)

1. Loader extracts pages/bytes (PDF text layer, TXT, DOCX).
2. **Digital text extraction** only.
3. Normalize Arabic text.
4. Chunk → embed (`src/models/bge-m3`) → upsert.
5. Hybrid retrieve → optional rerank.
6. Trust analysis marks RELEVANT indexes.
7. Grounded answer + Sources footer.

## Knowledge path

`decision_agent` → `query_planner` (Arabic READ-only plan) → `query_knowledge_base` against `KNOWLEDGE_COLLECTION`.

## Explicit search source (UI)

The React UI exposes a ChatGPT-style **Search source** control:

| UI | State flags | Effect |
|----|-------------|--------|
| **Auto** (default) | both false | Heuristics: document / web / KB / chat |
| **Database** | `use_knowledge_base=true` | Force KB path when no file is attached |
| **Search Internet** | `use_web_search=true` | Force live internet / Google-style search (beats document heuristics) |

If both flags are true, **Search Internet** wins. Database never overrides an uploaded file.

## Frontends

- **React** (`frontend/`): primary UI; LangGraph SDK streams runs; answer language + search source toggles.
- **Streamlit** (`streamlit_ui.py`): legacy local UI.
- **LangGraph API**: entry `src/agent/graph.py:graph` (`langgraph.json`).

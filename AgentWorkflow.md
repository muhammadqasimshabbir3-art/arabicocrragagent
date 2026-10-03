# Agent Workflow — Arabic Document Intelligence

## Graph

```text
START → prepare_input → decision_agent →
  query_documents
  | summarize_document
  | query_planner → query_knowledge_base
  | web_search
  | call_model
→ END
```

There is **no** `ingest_document` node. Upload handling (load → digital text extract → chunk → embed → index) runs inside `query_documents` / `summarize_document`, with an in-memory index cache. Text extraction is digital-only (no neural OCR).

## Routes

| Route | When | Pipeline |
|-------|------|----------|
| `query_documents` | Upload + question (Auto, or Database with file) | Spelling Guardian → ingest (cached) → hybrid retrieve top-k → optional rerank → trust analysis → grounded answer + Sources |
| `summarize_document` | Summarize intent / flag / auto-upload | ingest + summary |
| `query_planner` → `query_knowledge_base` | No upload; KB heuristic **or** UI **Database** on | Spelling Guardian → READ-only Arabic plan → hybrid retrieve against `KNOWLEDGE_COLLECTION` → trust → answer |
| `web_search` | Live / online heuristic **or** UI **Search Internet** on | DuckDuckGo (+ optional screenshots) → grounded chat |
| `call_model` | Greetings / general help / empty KB with Database on | Chat; does not invent document facts |

### UI search source (ChatGPT-style)

| Control | Input to graph | Priority |
|---------|----------------|----------|
| **Auto** | `use_web_search=false`, `use_knowledge_base=false` | Heuristics only |
| **Database** | `use_knowledge_base=true` | Force KB when no file; file still uses document path |
| **Search Internet** | `use_web_search=true` | Highest — always `web_search` |

**Spelling Guardian** (`subagents/spelling_corrector`): corrects only high-confidence Arabic/English typos before document/KB retrieval. Unsure tokens are left unchanged (`ENABLE_SPELLING_CORRECTOR`, `SPELLING_MIN_CONFIDENCE`).

## Chunking / storage knobs

- `CHUNK_STRATEGY`: `page_aware` (default), `recursive`, `layout_aware`, `semantic`
- `VECTORSTORE_BACKEND`: `chroma` (default), `faiss`, `qdrant`
- `ENABLE_RERANKER=true` with `RERANKER_BACKEND=lexical|cross_encoder`
- Optional metadata filters on retrieve: `page_start`, `page_end`, `filename`

## Grounding

1. Retrieve top-k passages (hybrid BM25 + BGE-M3 by default).
2. Trust layer: LLM marks each `[n]` RELEVANT / NOT_RELEVANT (JSON preferred, regex fallback).
3. Answer only from RELEVANT quotes, cite `[n]` and pages.
4. Append Sources footer listing all retrieved passages (USED vs reviewed).

## Language

`routing.language.detect_language` → `ar` | `en` | `mixed`. Prompts instruct the model to answer in the user’s language (or forced `response_language`) and keep evidence quotes untranslated.

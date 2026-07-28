# Agent Workflow — Arabic Document Intelligence

## Graph

```text
START → prepare_input → decision_agent →
  ingest_document | query_documents | summarize_document | call_model → END
```

## Routes

| Route | When | Pipeline |
|-------|------|----------|
| `ingest_document` | Upload without a focused question | load → OCR/digital → normalize → chunk → embed → index → short summary |
| `query_documents` | Upload + question | same ingest (cached) → hybrid retrieve top-k → optional rerank → trust analysis → grounded answer + Sources |
| `summarize_document` | Summarize intent / flag | ingest + summary |
| `call_model` | No document on this turn | General chat; does not invent document facts |

## Chunking / storage knobs

- `CHUNK_STRATEGY`: `page_aware` (default), `recursive`, `layout_aware`, `semantic`
- `VECTORSTORE_BACKEND`: `chroma` (default) or `faiss`
- `ENABLE_RERANKER=true` with `RERANKER_BACKEND=lexical|cross_encoder`
- Optional metadata filters on retrieve: `page_start`, `page_end`, `filename`

## Grounding

1. Retrieve top-k passages (hybrid BM25 + BGE-M3 by default).
2. Trust layer: LLM marks each `[n]` RELEVANT / NOT_RELEVANT with quotes.
3. Answer only from RELEVANT quotes, cite `[n]` and pages.
4. Append Sources footer listing all retrieved passages (USED vs reviewed).

## Language

`routing.language.detect_language` → `ar` | `en` | `mixed`. Prompts instruct the model to answer in the user’s language and keep evidence quotes untranslated.

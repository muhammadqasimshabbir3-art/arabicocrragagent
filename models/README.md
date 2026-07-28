# Local models (downloaded, not committed)

## What we use (free / open source)

| Role | Model / software | License | Source |
|------|------------------|---------|--------|
| **Vector DB** | **Chroma** (persistent under `data/chroma`) | Apache-2.0 | [chroma-core/chroma](https://github.com/chroma-core/chroma) |
| **Embeddings** | **BAAI/bge-m3** (1024-d, Arabic + English) | MIT | [Hugging Face](https://huggingface.co/BAAI/bge-m3) |
| **OCR (scanned)** | **Qari-OCR v0.3** `NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct` | Apache-2.0 weights | [Hugging Face](https://huggingface.co/NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct) |
| **Digital PDF text** | `pypdf` (no neural OCR) | BSD | skips Qari when a text layer exists |
| **LLM answers** | Groq `llama-3.1-8b-instant` | API (not local) | needs `GROQ_API_KEY` |

Qari is a **real local VLM** specialized for Arabic document OCR (layout-aware). It is among the strongest open Arabic OCR options vs classical engines on KITAB-Bench; AIN-7B is optional via `OCR_ENGINE=ain`.

## Embeddings (BGE-M3)

```bash
uv run python scripts/download_embedding_model.py
```

Creates `bge-m3/` — multilingual Arabic + English retrieval (~2 GB).

## OCR (Qari)

```bash
uv run python scripts/download_ocr_model.py
```

Or let the first **scanned** document ingest download
`NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct` automatically (~4–5 GB).

## Seed a free Arabic sample into Chroma

```bash
uv run python scripts/make_arabic_sample_pdf.py   # MIT sample circular
uv run python scripts/seed_arabic_sample.py       # embed → Chroma → retrieve
uv run python scripts/seed_arabic_sample.py --answer --ask "ما مدة الإجازة السنوية؟"
```

Vectors land in `data/chroma` (gitignored). Upload the same PDF in the React UI to ask interactively.

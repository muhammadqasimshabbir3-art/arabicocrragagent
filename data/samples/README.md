# Sample documents (free / open source)

| File | License | Notes |
|------|---------|--------|
| `arabic_admin_circular.pdf` | MIT (authored in-repo) | Small 3-page smoke PDF — `scripts/make_arabic_sample_pdf.py` |
| `pdfs/arabic_sample_200lines.pdf` | MIT (authored in-repo) | Standard digital RAG evaluation sample (~200 lines) |
| `pdfs/arabic_bayan_text.pdf` | Public domain (Internet Archive) | Text-layer Arabic book sample |

Generate / refresh:

```bash
uv run python scripts/make_arabic_sample_pdf.py
uv run python scripts/make_arabic_large_sample_pdf.py --lines 200
uv run python scripts/seed_arabic_sample.py
uv run python scripts/seed_arabic_sample.py --pdf data/samples/pdfs/arabic_sample_200lines.pdf --no-knowledge --ask "ما مدة الإجازة السنوية؟" --answer
```

The seed script embeds with **BGE-M3** (`src/models/bge-m3`), stores vectors in Chroma (`src/data/chroma` by default), and retrieves sample questions.

This agent is **digital-text only** — use text-layer PDFs / TXT / DOCX.

## Alf Layla knowledge push (Qdrant) + checks

```bash
export VECTORSTORE_BACKEND=qdrant

PYTHONUNBUFFERED=1 uv run python scripts/reseed_alf_layla_knowledge.py \
  --chars 120000 --batch 8 --test-answer

PYTHONUNBUFFERED=1 uv run python scripts/eval_retrieval_100_kb.py
PYTHONUNBUFFERED=1 uv run python scripts/eval_retrieval_100_challenge.py --delay 0.5
```

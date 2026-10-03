# Release checklist (v1.x)

## Pre-release

- [ ] Bump `pyproject.toml` / README version.
- [ ] Env templates: `.env.example`, `src/deploy/.env.railway.example`, `.env.vercel.example`.
- [ ] Docs aligned; no Qari / heavy OCR references as product path.
- [ ] Models path = `src/models/`; Dockerfile = `src/Dockerfile`.
- [ ] `langgraph.json` has no `http.cors` override.

## Quality

- [ ] `uv sync --group dev`
- [ ] `uv run ruff check src tests`
- [ ] `uv run pytest tests/unit_tests -q`
- [ ] Smoke: text-layer PDF upload + KB question.

## Deploy

- [ ] Railway: `src/Dockerfile` build; volume on `/deps/ArabicOCRRAGAgent/src`; `/ok` green.
- [ ] Railway vars from `src/deploy/.env.railway.example`.
- [ ] Vercel: absolute `VITE_LANGGRAPH_API_URL`; redeploy after `VITE_*` changes.
- [ ] CORS includes Vercel origin.
